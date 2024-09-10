import gradio as gr
import os
import threading
import queue
from datetime import datetime
import csv
import webbrowser
from model.trainer import train_model
from telegram_backend.bot import launch_bot
from config import global_config, save_config, load_config

current_dir = os.path.dirname(os.path.abspath(__file__))
action_queue = queue.Queue()
console_output = []

def get_model_stats():
    csv_path = os.path.join(current_dir, 'data', 'models', 'model_stats.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)
            return {
                'datetime': datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'),
                'accuracy': float(row[1]),
                'images_trained': int(row[2])
            }
    return None

def custom_print(*args, sep=' ', end='\n'):
    message = sep.join(map(str, args)) + end
    console_output.append(message)
    return "\n".join(console_output)

def run_training():
    action_queue.put(("train", None))
    return "Training queued. Check console for progress."

def start_bot():
    action_queue.put(("bot", None))
    return "Bot start queued. Check console for progress."

def create_me621_interface():
    config_loaded = load_config()

    def save_config_and_update_ui(username, api_key, chat_id, bot_api):
        save_config(username, api_key, chat_id, bot_api)
        return "Configuration saved successfully!", gr.update(visible=True), gr.update(visible=False)

    def toggle_visibility(show_main=True, show_config=False, show_annotation=False):
        return (
            gr.update(visible=show_main),
            gr.update(visible=show_config),
            gr.update(visible=show_annotation)
        )

    with gr.Blocks(title="ME621") as demo:
        main_container = gr.Column(visible=config_loaded)
        config_container = gr.Column(visible=not config_loaded)
        annotation_container = gr.Column(visible=False)


        with config_container:
            gr.Markdown("# ME621 Configuration")
            config_inputs = [
                gr.Textbox(label="e621 Username", value=global_config.get('username', '')),
                gr.Textbox(label="e621 API Key", type="password", value=global_config.get('api_key', '')),
                gr.Textbox(label="Telegram Chat ID", value=global_config.get('chat_id', '')),
                gr.Textbox(label="Telegram API Key", type="password", value=global_config.get('bot_api', ''))
            ]
            config_btn = gr.Button("Save Configuration")

        with main_container:
            gr.Markdown("# ME621")
            
            model_stats = get_model_stats()
            if model_stats:
                model_info = f"""
                <p>Current model: {model_stats['accuracy']:.2f}% accuracy</p>
                <p>Trained on {model_stats['images_trained']} images</p>
                <p>Created: {model_stats['datetime'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                """
            else:
                model_info = "<p>No model trained</p>"
            gr.HTML(model_info)

            with gr.Row():
                annotation_btn = gr.Button("Annotation")
                train_btn = gr.Button("Start Training")
                bot_btn = gr.Button("Launch Bot")

            output = gr.Textbox(label="Console Output", interactive=False)

            with gr.Accordion("Update Configuration", open=False):
                update_config_inputs = [
                    gr.Textbox(label="e621 Username", value=global_config.get('username', '')),
                    gr.Textbox(label="e621 API Key", type="password", value=global_config.get('api_key', '')),
                    gr.Textbox(label="Telegram Chat ID", value=global_config.get('chat_id', '')),
                    gr.Textbox(label="Telegram API Key", type="password", value=global_config.get('bot_api', ''))
                ]
                update_config_btn = gr.Button("Update Configuration")
        
        with annotation_container:
            with gr.Row():
                help_button = gr.Button("Help", scale=1)
                gr.Button("Back to Main Menu", scale=1).click(
                    lambda: toggle_visibility(True, False, False),
                    outputs=[main_container, config_container, annotation_container]
                )
                
            with gr.Column():
                help_text = gr.HTML(visible=False, value="""
                <ul>
                    <li>Select any image you find appealing and want to see more of</li>
                    <li>Try to be as consistent as possible</li>
                    <li>Then click 'Save and Next'</li>
                    <li>Clicking the heart will favorite the image on your e621 account</li>
                    <li>We recommend labeling at least 500 good images and 5,000 bad images</li>
                </ul>
                """)
                help_button.click(lambda: gr.update(visible=True), outputs=help_text)

            
            loading_bar = gr.Progress()
            image_grid = gr.Gallery(label="Image Grid", show_label=False, elem_id="image-grid", columns=4, rows=4, height="auto")
            save_button = gr.Button("Save and Next")

            def load_images():
                from e621_backend.annotation import fetch_and_process_images
                return fetch_and_process_images()

            demo.load(load_images, outputs=[loading_bar, image_grid])

            # Placeholder for image grid
            with gr.Column():
                gr.Markdown("## Image Grid")
                with gr.Row():
                    for _ in range(4):  # Assuming a 4x4 grid
                        with gr.Column():
                            for _ in range(4):
                                gr.Image(label="Image", interactive=True)

            gr.Button("Save and Next")


        
        # Event handlers
        
        annotation_btn.click(
            lambda: toggle_visibility(False, False, True),
            outputs=[main_container, config_container, annotation_container]
        )
                
        config_btn.click(
            save_config_and_update_ui,
            inputs=config_inputs,
            outputs=[output, main_container, config_container]
        )

        update_config_btn.click(
            save_config,
            inputs=update_config_inputs,
            outputs=output
        )

        train_btn.click(run_training, outputs=output)
        bot_btn.click(start_bot, outputs=output)
        
        demo.load(
            lambda: (gr.update(visible=config_loaded), gr.update(visible=not config_loaded)),
            outputs=[main_container, config_container]
        )

    return demo

def launch_me621():
    demo = create_me621_interface()

    def launch_interface():
        demo.launch(prevent_thread_lock=True)
        webbrowser.open('http://127.0.0.1:7860')

    threading.Thread(target=launch_interface, daemon=True).start()

    bot_instance = None
    try:
        while True:
            if not action_queue.empty():
                action, _ = action_queue.get()
                if action == "train":
                    train_model(custom_print)
                elif action == "bot":
                    if bot_instance is None:
                        bot_instance = launch_bot(custom_print)
                    else:
                        custom_print("Bot is already running.")
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        if bot_instance:
            bot_instance.stop()

if __name__ == "__main__":
    launch_me621()