from flask import Flask, render_template, request, jsonify
import os

from ingest import process_single_pdf
import rag  # import the module, not just functions
           # so reload_db() updates the SAME globals ask() reads

app = Flask(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# =========================
# Upload Page
# =========================
@app.route("/")
@app.route("/upload")
def upload_page():
    return render_template("upload.html")


# =========================
# Upload API
# =========================
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]

        if file.filename == "":
            return "Empty filename", 400

        file_path = os.path.join(DATA_DIR, file.filename)
        file.save(file_path)

        print("📄 File saved:", file.filename)

        # Ingest the new PDF → writes new index.faiss + chunks.txt to disk
        process_single_pdf(file_path)

        # Hot-reload: update index + chunks in THIS process's memory
        rag.reload_db()

        print(f"[APP] Reload done — {len(rag.chunks)} chunks now in memory")

        return "✅ PDF uploaded and indexed!"

    except Exception as e:
        print("❌ Upload Error:", e)
        return "Upload failed", 500


# =========================
# Chat Page
# =========================
@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")


# =========================
# Chat API
# =========================
@app.route("/chat", methods=["POST"])
def chat_api():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"answer": "Invalid request"}), 400

        query = data.get("query", "")

        if not query:
            return jsonify({"answer": "Empty query"}), 400

        print("🧑 Query:", query)

        answer = rag.ask(query)

        print("🤖 Answer:", answer)

        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ Chat Error:", e)
        return jsonify({"answer": "Server error"}), 500


@app.route("/ask", methods=["POST"])
def ask_api():
    data = request.get_json()
    query = data.get("query")
    answer = rag.ask(query)
    return jsonify({"answer": answer})


# =========================
# Run App
# =========================
if __name__ == "__main__":
    # use_reloader=False prevents Werkzeug from spawning a child process.
    # With use_reloader=True (default in debug mode), Flask runs TWO processes:
    # the reloader parent and a worker child. reload_db() would update globals
    # in the child, but the parent's copy stays stale — causing exactly the
    # symptom you saw. Disabling the reloader means one process, one memory
    # space, and reload_db() always takes effect immediately.
    app.run(host="0.0.0.0", port=9005, debug=True, use_reloader=False)
