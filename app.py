import gradio as gr
from inference import pipeline

def classify_email(email_text):
    result = pipeline({"text": email_text})
    
    if "error" in result:
        return result["error"], "", ""
    
    return (
        result["verdict"],
        f"{result['confidence']*100:.2f}%",
        result["reasoning"]
    )

iface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=7, label="Enter Email Text"),
    outputs=[
        gr.Textbox(label="Verdict (SPAM / NOT SPAM)"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Reasoning")
    ],
    title="Spam Classifier & Explainer",
    description="This tool classifies emails as spam or not and provides reasoning using similar past examples."
)

if __name__ == "__main__":
    iface.launch()
