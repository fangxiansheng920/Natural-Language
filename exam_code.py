import jieba
import jieba.posseg as pseg
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, Image


# ==================== NLP核心处理模块 ====================
def load_text(filepath):
    """读取文本文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def tokenize(text, user_dict=None):
    """分词功能"""
    if user_dict:
        jieba.load_userdict(user_dict)
    return list(jieba.cut(text))


def get_word_freq(tokens, stopwords_path='stopwords.txt'):
    """词频统计"""
    try:
        stopwords = set(open(stopwords_path, encoding='utf-8').read().split())
    except FileNotFoundError:
        stopwords = set()
    return Counter([w for w in tokens if w not in stopwords and len(w) > 1])


def pos_analysis(text, save_path='pos_result.txt'):
    """词性标注并保存"""
    words = pseg.cut(text)
    result = [f"{word} {flag}" for word, flag in words]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
    return result


def extract_entities(pos_result, entity_tags):
    """提取特定实体"""
    return [(word.split()[0], word.split()[1])
            for word in pos_result
            if word.split()[1] in entity_tags]


def generate_wordcloud(text, save_path='wordcloud.png'):
    """生成词云"""
    wc = WordCloud(font_path='msyh.ttc', width=800, height=600)
    wc.generate(text)
    wc.to_file(save_path)
    return save_path


def plot_freq_chart(counter, top_n=10, chart_type='bar', save_path='chart.png'):
    """生成频率图表"""
    # 设置中文字体为黑体，解决中文标签显示乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决坐标轴负号显示异常问题
    plt.rcParams['axes.unicode_minus'] = False
    plt.switch_backend('Agg')  # 解决GUI中的线程问题
    items = counter.most_common(top_n)
    words, counts = zip(*items)

    plt.figure(figsize=(12, 6))
    if chart_type == 'bar':
        plt.bar(words, counts)
    elif chart_type == 'pie':
        plt.pie(counts, labels=words, autopct='%1.1f%%')

    plt.title('词频分布图')
    plt.savefig(save_path)
    plt.close()
    return save_path


def manage_custom_dict(word, action='add', dict_path='user_dict.txt'):
    """管理用户词典"""
    if action == 'add':
        with open(dict_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{word}")
    elif action == 'remove':
        lines = [line for line in open(dict_path, encoding='utf-8')
                 if line.strip() != word]
        with open(dict_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


# ==================== GUI界面模块 ====================
class NLPApp:
    def __init__(self, master):
        self.master = master
        master.title("自然语言处理系统")
        master.geometry("1000x700")

        # 初始化属性
        self.raw_text = ""
        self.tokens = []
        self.word_freq = None

        # 界面布局
        self.create_file_frame()
        self.create_analysis_frame()
        self.create_visual_frame()
        self.create_output_frame()

    def create_file_frame(self):
        """文件选择区域"""
        frame = ttk.LabelFrame(self.master, text="文件管理")
        frame.pack(fill='x', padx=10, pady=5)

        self.filepath = tk.StringVar()
        ttk.Entry(frame, textvariable=self.filepath, width=50).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text="选择文件", command=self.load_file).grid(row=0, column=1)
        ttk.Button(frame, text="加载自定义词典", command=self.load_dict).grid(row=0, column=2)

    def create_analysis_frame(self):
        """分析功能区域"""
        frame = ttk.LabelFrame(self.master, text="分析功能")
        frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(frame, text="执行分词", command=self.run_tokenize).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text="词频统计", command=self.run_word_freq).grid(row=0, column=1)
        ttk.Button(frame, text="词性分析", command=self.run_pos_analysis).grid(row=0, column=2)
        ttk.Button(frame, text="实体抽取", command=self.run_entity_extract).grid(row=0, column=3)

    def create_visual_frame(self):
        """可视化区域"""
        frame = ttk.LabelFrame(self.master, text="可视化")
        frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.canvas_frame = ttk.Frame(frame)
        self.canvas_frame.pack(side='left', fill='both', expand=True)

        self.img_label = ttk.Label(self.canvas_frame)
        self.img_label.pack()

        control_frame = ttk.Frame(frame)
        control_frame.pack(side='right', padx=10)
        ttk.Button(control_frame, text="生成词云", command=self.show_wordcloud).pack(pady=5)
        ttk.Button(control_frame, text="柱状图",
                   command=lambda: self.show_chart('bar')).pack(pady=5)
        ttk.Button(control_frame, text="饼状图",
                   command=lambda: self.show_chart('pie')).pack(pady=5)

    def create_output_frame(self):
        """结果显示区域"""
        frame = ttk.LabelFrame(self.master, text="分析结果")
        frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.result_text = tk.Text(frame, wrap='word')
        scrollbar = ttk.Scrollbar(frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    # 核心功能方法
    def load_file(self):
        """加载文本文件"""
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.filepath.set(path)
            try:
                self.raw_text = load_text(path)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "文件加载成功！")
            except Exception as e:
                messagebox.showerror("错误", f"文件读取失败：{str(e)}")

    def load_dict(self):
        """加载自定义词典"""
        path = filedialog.askopenfilename(filetypes=[("Dict files", "*.txt")])
        if path:
            try:
                jieba.load_userdict(path)
                messagebox.showinfo("提示", "自定义词典加载成功！")
            except Exception as e:
                messagebox.showerror("错误", f"词典加载失败：{str(e)}")

    def run_tokenize(self):
        """执行分词"""
        if not hasattr(self, 'raw_text') or not self.raw_text:
            messagebox.showerror("错误", "请先选择文件！")
            return
        try:
            self.tokens = tokenize(self.raw_text)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "分词结果：\n" + "/".join(self.tokens[:200]) + "...")
        except Exception as e:
            messagebox.showerror("错误", f"分词失败：{str(e)}")

    def run_word_freq(self):
        """词频统计"""
        if not self.tokens:
            messagebox.showerror("错误", "请先执行分词！")
            return
        try:
            self.word_freq = get_word_freq(self.tokens)
            self.result_text.delete(1.0, tk.END)
            top_words = self.word_freq.most_common(10)
            self.result_text.insert(tk.END, "词频统计结果（Top 10）：\n")
            for word, count in top_words:
                self.result_text.insert(tk.END, f"{word}: {count}\n")
        except Exception as e:
            messagebox.showerror("错误", f"词频统计失败：{str(e)}")

    def run_pos_analysis(self):
        """词性分析"""
        if not self.raw_text:
            messagebox.showerror("错误", "请先加载文件！")
            return
        try:
            pos_result = pos_analysis(self.raw_text)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "词性分析结果已保存到 pos_result.txt")
        except Exception as e:
            messagebox.showerror("错误", f"词性分析失败：{str(e)}")

    def run_entity_extract(self):
        """实体抽取"""
        if not self.raw_text:
            messagebox.showerror("错误", "请先加载文件！")
            return
        try:
            pos_result = pseg.cut(self.raw_text)
            entities = extract_entities([f"{w} {f}" for w, f in pos_result], ['nr', 'ns'])
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "抽取的实体：\n" + "\n".join([f"{w} ({t})" for w, t in entities]))
        except Exception as e:
            messagebox.showerror("错误", f"实体抽取失败：{str(e)}")

    def show_wordcloud(self):
        """显示词云"""
        if not self.raw_text:
            messagebox.showerror("错误", "请先加载文件！")
            return
        try:
            img_path = generate_wordcloud(" ".join(self.tokens))
            self.display_image(img_path)
        except Exception as e:
            messagebox.showerror("错误", f"生成词云失败：{str(e)}")

    def show_chart(self, chart_type):
        """显示图表"""
        if not self.word_freq:
            messagebox.showerror("错误", "请先进行词频统计！")
            return
        try:
            img_path = plot_freq_chart(self.word_freq, chart_type=chart_type)
            self.display_image(img_path)
        except Exception as e:
            messagebox.showerror("错误", f"生成图表失败：{str(e)}")

    def display_image(self, img_path):
        """显示图片"""
        try:
            img = Image.open(img_path)
            img = img.resize((600, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"图片显示失败：{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = NLPApp(root)
    root.mainloop()