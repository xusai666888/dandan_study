"""
SpongeBob 模型交互式对话脚本 (最终修复版)
修复点：
1. 移除不存在的 stop_strings 参数，改用生成后截断。
2. 移除 TextStreamer，避免输出冲突，实现精准控制。
3. 修复 Pretrain 模式下的历史记忆逻辑。
4. 增加强制停止标记检测，防止模型自动续写多轮。
"""
import argparse
import torch
import re
from transformers import AutoTokenizer
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM

def main():
    parser = argparse.ArgumentParser(description="SpongeBob模型交互对话")
    parser.add_argument('--model_path', default='/root/autodl-tmp/dandan_study/pretrain_out/exp_1/h768_l12_bs128_lr0.001/global_step_46229/pretrain_768.pth', type=str)
    parser.add_argument('--tokenizer_path', default='./tokenizer_15k', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=12, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int, help="单次最大生成长度，设小一点防止啰嗦")
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.8, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--multi_turn', action='store_true', help="开启多轮对话记忆")
    args = parser.parse_args()

    # 自动推断类型
    model_type = 'pretrain' if 'pretrain' in args.model_path else 'sft'
    
    print(f'正在加载模型：{args.model_path}')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    except Exception:
        # 兼容本地 tokenizer 可能没有 trust_remote_code 的情况
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 确保 eos_token 存在，防止生成无限循环
    if tokenizer.eos_token is None:
        tokenizer.eos_token = '</s>'
        if '</s>' not in tokenizer.get_vocab():
            # 如果词表里真没有，临时加一个（虽然不太可能）
            tokenizer.add_special_tokens({'eos_token': '</s>'})
    
    model = SpongeBobForCausalLM(SpongeBobConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    ))
    
    # 加载权重
    state_dict = torch.load(args.model_path, map_location=args.device)
    # 处理可能的 key 不匹配问题 (如果有 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.eval().to(args.device)
    
    print(f'✅ 模型加载完成 | 设备：{args.device} | 类型：{model_type}')
    print(f'💡 提示：输入 "exit" 退出。默认单轮回复，如需多轮请添加 --multi_turn 参数')
    print('='*60)

    # 用于存储历史对话的列表 (纯文本格式，方便 pretrain 和 sft 通用处理)
    history_text = ""

    while True:
        try:
            user_input = input('\n👤 你：').strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        
        if user_input.lower() in ['exit', 'quit', '退出', 'q']:
            print('👋 再见！')
            break
        if not user_input:
            continue

        # --- 构建 Prompt ---
        if model_type == 'sft':
            # SFT 模式：使用 Chat Template
            if args.multi_turn:
                # 需要从 history_text 还原成 messages 列表比较麻烦，这里简化处理：
                # 如果是多轮，我们简单地把历史拼接到 prompt 里，或者重新维护一个 messages 列表
                # 为了稳健，这里我们重新维护一个 messages 列表
                if 'messages' not in locals():
                    messages = []
                messages.append({"role": "user", "content": user_input})
                
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except:
                    # fallback
                    prompt = f"你：{user_input}\nSpongeBob: "
            else:
                messages = [{"role": "user", "content": user_input}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except:
                    prompt = f"你：{user_input}\nSpongeBob: "
        
        else:
            # Pretrain 模式：手动构造文本
            # 格式：... \n你：xxx \nSpongeBob: 
            current_turn = f"你：{user_input}\nSpongeBob: "
            
            if args.multi_turn:
                # 拼接历史
                prompt = history_text + current_turn
            else:
                prompt = current_turn
                history_text = "" # 单轮模式清空历史

        # --- Tokenize ---
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        input_len = inputs['input_ids'].shape[1]

        # --- Generate ---
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
                # 注意：这里不再传 stop_strings，因为原生不支持，我们在后面截断
            )
        
        # --- Decode & Truncate (关键步骤) ---
        generated_ids = outputs[0][input_len:]
        full_response = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 【核心修复】手动截断，防止模型自问自答
        # 1. 遇到换行符就停 (假设模型说完一句会换行)
        stop_markers = ['\n', '你：', 'User:', 'A:', 'B:', '<|im_end|>', '</s>']
        
        # 查找第一个出现的停止标记的位置
        first_stop_pos = len(full_response)  # 默认为整个响应的长度
        for marker in stop_markers:
            pos = full_response.find(marker)
            if pos != -1 and pos < first_stop_pos:
                first_stop_pos = pos
        
        # 截断响应
        response = full_response[:first_stop_pos].strip()
        
        # 打印模型回复
        print(f'🧽 SpongeBob: {response}')

        # 更新历史（仅在 multi_turn 模式下）
        if args.multi_turn:
            history_text += f"{current_turn}{response}\n"

if __name__ == "__main__":
    main()