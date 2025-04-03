import gradio as gr
import subprocess
import os
import time
import fcntl
from typing import List, Tuple

class LLMInterface:
    def __init__(self):
        self.chat_history: List[Tuple[str, str]] = []
        self.chat_process = None

    def start_chat(
        self,
        model_path: str,
        tokenizer_path: str,
        use_cuda: bool,
        is_quantized: bool,
        max_length: int = 1024,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        try:
            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                return "错误: 模型或分词器文件不存在"
                
            if self.chat_process is not None:
                self.stop_chat()

            cmd = [
                "./web/bin/qwen_chat",
                model_path,
                tokenizer_path,
                str(is_quantized).lower(),
                str(use_cuda).lower(),
                str(max_length),
                system_prompt,
            ]

            self.chat_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            
            # 读取初始化输出
            stdout_fd = self.chat_process.stdout.fileno()
            stdout_flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
            fcntl.fcntl(stdout_fd, fcntl.F_SETFL, stdout_flags | os.O_NONBLOCK)
            time.sleep(1)
            
            try:
                initial_output = self.chat_process.stdout.read()
            except:
                pass
            finally:
                fcntl.fcntl(stdout_fd, fcntl.F_SETFL, stdout_flags)
            
            self.chat_history = []
            return "聊天进程已启动"

        except Exception as e:
            return f"启动失败: {str(e)}"

    def send_message(
        self,
        message: str,
        history: List[Tuple[str, str]],
    ) -> Tuple[str, List[Tuple[str, str]]]:
        try:
            if self.chat_process is None or self.chat_process.poll() is not None:
                return "错误: 聊天进程未启动或已结束", history
                
            self.chat_process.stdin.write(message + "\n")
            self.chat_process.stdin.flush()
            
            # 设置非阻塞读取
            stdout_fd = self.chat_process.stdout.fileno()
            stderr_fd = self.chat_process.stderr.fileno()
            stdout_flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
            stderr_flags = fcntl.fcntl(stderr_fd, fcntl.F_GETFL)
            fcntl.fcntl(stdout_fd, fcntl.F_SETFL, stdout_flags | os.O_NONBLOCK)
            fcntl.fcntl(stderr_fd, fcntl.F_SETFL, stderr_flags | os.O_NONBLOCK)
            
            output = ""
            start_time = time.time()
            
            while time.time() - start_time < 30:
                try:
                    chunk = self.chat_process.stdout.read()
                    if chunk:
                        output += chunk
                    if "[生成时间:" in output and "请输入问题:" in output:
                        break
                except:
                    pass
                time.sleep(0.1)
            
            fcntl.fcntl(stdout_fd, fcntl.F_SETFL, stdout_flags)
            fcntl.fcntl(stderr_fd, fcntl.F_SETFL, stderr_flags)
            
            # 提取回复
            try:
                assistant_start = output.find("助手:") + 3
                generation_time_start = output.find("[生成时间:")
                
                if assistant_start > 3 and generation_time_start > assistant_start:
                    response = output[assistant_start:generation_time_start].strip()
                    if response:
                        history.append((message, response))
                        return response, history
                
                return "无法解析助手回复", history
                
            except Exception as e:
                return f"处理回复时出错: {str(e)}", history

        except Exception as e:
            return f"发送消息时出错: {str(e)}", history

    def stop_chat(self) -> str:
        if self.chat_process is None:
            return "聊天进程未启动"
            
        try:
            self.chat_process.terminate()
            try:
                self.chat_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.chat_process.kill()
                self.chat_process.wait()
            
            self.chat_process = None
            return "聊天进程已停止"
        except Exception as e:
            return f"停止失败: {str(e)}"

    def text_completion(
        self,
        model_path: str,
        tokenizer_path: str,
        use_cuda: bool,
        is_quantized: bool,
        prompt: str,
        max_length: int = 1024,
    ) -> str:
        """文本补全功能"""
        try:
            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                return "错误: 模型或分词器文件不存在"

            cmd = [
                "./web/bin/llama_gen",
                model_path,
                tokenizer_path,
                str(is_quantized).lower(),
                str(use_cuda).lower(),
                str(max_length),
                prompt,
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            try:
                output, error = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                return "错误: 命令执行超时"

            if error and process.returncode != 0:
                return f"错误: {error}"

            return output.strip() if output.strip() else "没有生成结果"

        except Exception as e:
            return f"发生错误: {str(e)}"

def create_interface():
    interface = LLMInterface()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def send_and_clear(message, history):
        if not message.strip():
            return "", history
        response, updated_history = interface.send_message(message, history)
        return "", updated_history

    with gr.Blocks(
        title="大模型推理界面",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown("# 大模型推理界面")

        # 通用配置部分
        with gr.Row():
            with gr.Column(scale=1):
                model_path = gr.Textbox(
                    label="模型文件路径",
                    value=os.path.join(current_dir, "model.bin"),
                    container=False,
                )
                use_cuda = gr.Checkbox(
                    label="使用CUDA加速",
                    value=True,
                    container=False,
                )
            with gr.Column(scale=1):
                tokenizer_path = gr.Textbox(
                    label="分词器文件路径",
                    value=os.path.join(current_dir, "tokenizer.json"),
                    container=False,
                )
                is_quantized = gr.Checkbox(
                    label="使用量化模型",
                    value=False,
                    container=False,
                )

        # 最大生成长度（共用）
        max_length = gr.Slider(
            minimum=64,
            maximum=2048,
            value=1024,
            step=64,
            label="最大生成长度",
        )

        with gr.Tabs() as tabs:
            # 文本补全标签页
            with gr.Tab("文本补全") as completion_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="输入提示词",
                            placeholder="请输入要补全的文本...",
                            lines=3,
                        )
                        complete_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=1):
                        completion_output = gr.Textbox(
                            label="生成结果",
                            lines=10,
                            interactive=False,
                        )

                complete_btn.click(
                    fn=interface.text_completion,
                    inputs=[
                        model_path,
                        tokenizer_path,
                        use_cuda,
                        is_quantized,
                        prompt_input,
                        max_length,
                    ],
                    outputs=completion_output,
                )

            # 对话标签页
            with gr.Tab("对话") as chat_tab:
                with gr.Row():
                    system_prompt = gr.Textbox(
                        label="系统提示词",
                        value="You are a helpful assistant.",
                        lines=2,
                    )
                    chat_status = gr.Textbox(label="状态", value="未启动", interactive=False)

                chatbot = gr.Chatbot(label="对话历史", height=400)
                
                msg = gr.Textbox(label="输入消息", lines=3)
                
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear = gr.Button("清除对话", variant="secondary", scale=1)

                # 绑定事件处理
                send_btn.click(fn=send_and_clear, inputs=[msg, chatbot], outputs=[msg, chatbot])
                msg.submit(fn=send_and_clear, inputs=[msg, chatbot], outputs=[msg, chatbot])
                clear.click(fn=lambda: None, outputs=chatbot)

        # 添加标签页切换事件处理
        chat_tab.select(
            fn=lambda *args: interface.start_chat(*args) if interface.stop_chat() else "切换失败",
            inputs=[model_path, tokenizer_path, use_cuda, is_quantized, max_length, system_prompt],
            outputs=chat_status,
        )
        
        completion_tab.select(
            fn=interface.stop_chat,
            outputs=chat_status,
        )

        return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)