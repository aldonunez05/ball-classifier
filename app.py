"""
Sports Ball Classifier - Web App
=================================
pip install flask pillow torch torchvision
python app.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64

app = Flask(__name__)

CLASS_NAMES = [
    "Baseball", "Basketball", "Billiards", "Bowling", "Cricket",
    "Football", "Golf", "Rugby", "Tennis", "Volleyball"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Load model once at startup ────────────────────────────────────────────────

def build_resnet(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

model = build_resnet(len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model_resnet.pth", map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── HTML (single page, no CSS framework) ─────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html>
<head><title>Ball Classifier</title></head>
<body>
  <h2>Sports Ball Classifier</h2>
  <p>Drag and drop an image below, or click to select one.</p>

  <div id="drop" style="width:300px;height:150px;border:2px dashed #000;
       display:flex;align-items:center;justify-content:center;cursor:pointer;
       user-select:none">
    Drop image here or click
  </div>

  <input type="file" id="fileinput" accept="image/*" style="display:none">

  <br>
  <img id="preview" style="max-width:300px;display:none"><br><br>
  <div id="result"></div>

  <script>
    const drop = document.getElementById("drop");
    const fileinput = document.getElementById("fileinput");

    // click to open file picker
    drop.addEventListener("click", () => fileinput.click());
    fileinput.addEventListener("change", () => handleFile(fileinput.files[0]));

    // drag and drop
    drop.addEventListener("dragenter", e => { e.preventDefault(); drop.style.background = "#eee"; });
    drop.addEventListener("dragover",  e => { e.preventDefault(); drop.style.background = "#eee"; });
    drop.addEventListener("dragleave", e => { drop.style.background = ""; });
    drop.addEventListener("drop", e => {
      e.preventDefault();
      drop.style.background = "";

      // Case 1: dragging a file from your computer
      const file = e.dataTransfer.files[0];
      if (file) { handleFile(file); return; }

      // Case 2: dragging an image from a browser tab (gives a URL, not a file)
      const url = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
      if (url && url.startsWith("http")) { handleURL(url); return; }

      document.getElementById("result").innerText = "Could not read the dropped item.";
    });

    function showResult(data) {
      if (data.error) {
        document.getElementById("result").innerText = "Error: " + data.error;
        return;
      }
      let html = "<b>Prediction: " + data.prediction + "</b><br><br>Top 3:<br>";
      data.top3.forEach(([cls, pct]) => { html += cls + ": " + pct + "%<br>"; });
      document.getElementById("result").innerHTML = html;
    }

    // dropped from desktop — send as file upload
    function handleFile(file) {
      if (!file || !file.type.startsWith("image/")) {
        document.getElementById("result").innerText = "Please drop an image file.";
        return;
      }
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.getElementById("preview");
        img.src = e.target.result;
        img.style.display = "block";
      };
      reader.readAsDataURL(file);

      const form = new FormData();
      form.append("image", file);
      document.getElementById("result").innerText = "Classifying...";
      fetch("/predict", { method: "POST", body: form })
        .then(r => r.json()).then(showResult)
        .catch(err => { document.getElementById("result").innerText = "Request failed: " + err; });
    }

    // dropped from browser tab — load into canvas first to get a reliable data URI
    function handleURL(url) {
      document.getElementById("result").innerText = "Loading image...";
      const imgEl = document.getElementById("preview");
      const tempImg = new Image();
      tempImg.crossOrigin = "anonymous";
      tempImg.onload = function() {
        // draw onto canvas to convert to data URI regardless of original format
        const canvas = document.createElement("canvas");
        canvas.width = tempImg.naturalWidth;
        canvas.height = tempImg.naturalHeight;
        canvas.getContext("2d").drawImage(tempImg, 0, 0);
        const dataURL = canvas.toDataURL("image/jpeg");

        imgEl.src = dataURL;
        imgEl.style.display = "block";
        document.getElementById("result").innerText = "Classifying...";

        fetch("/predict_url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: dataURL })
        })
          .then(r => r.json()).then(showResult)
          .catch(err => { document.getElementById("result").innerText = "Request failed: " + err; });
      };
      tempImg.onerror = function() {
        // canvas approach failed (CORS) — fall back to sending the raw URL
        imgEl.src = url;
        imgEl.style.display = "block";
        document.getElementById("result").innerText = "Classifying...";
        fetch("/predict_url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: url })
        })
          .then(r => r.json()).then(showResult)
          .catch(err => { document.getElementById("result").innerText = "Request failed: " + err; });
      };
      tempImg.src = url;
    }
  </script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"})

    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]

        top3_probs, top3_idx = probs.topk(3)
        prediction = CLASS_NAMES[top3_idx[0].item()]
        top3 = [
            (CLASS_NAMES[i.item()], round(p.item() * 100, 1))
            for i, p in zip(top3_idx, top3_probs)
        ]
        return jsonify({"prediction": prediction, "top3": top3})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/predict_url", methods=["POST"])
def predict_url():
    import urllib.request
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "no url provided"})

    # Some browsers pass the src of an <img> tag which may be a data URI
    if url.startswith("data:image"):
        try:
            header, b64data = url.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"error": "could not decode data URI: " + str(e)})
    else:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/webp,image/apng,image/*,*/*"
            })
            response = urllib.request.urlopen(req, timeout=10)
            content_type = response.headers.get("Content-Type", "")
            img_bytes = response.read()

            # If the server returned HTML instead of an image, tell the user
            if "text/html" in content_type:
                return jsonify({"error": "URL points to a webpage, not an image. Try right-clicking the image and copying the image address instead."})

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"error": "could not fetch image: " + str(e)})

    try:
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        top3_probs, top3_idx = probs.topk(3)
        prediction = CLASS_NAMES[top3_idx[0].item()]
        top3 = [
            (CLASS_NAMES[i.item()], round(p.item() * 100, 1))
            for i, p in zip(top3_idx, top3_probs)
        ]
        return jsonify({"prediction": prediction, "top3": top3})
    except Exception as e:
        return jsonify({"error": "model error: " + str(e)})


if __name__ == "__main__":
    print(f"Model loaded on {device}")
    print("Open http://localhost:5000")
    app.run(debug=False)
