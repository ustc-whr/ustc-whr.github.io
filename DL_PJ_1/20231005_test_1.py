from bs4 import BeautifulSoup
import requests
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from main_window import Ui_MainWindow

def fetch_articles_by_author(name):
    base_url = "https://dblp.org"
    query = f'/search?q={name.replace(" ", "_")}'
    full_url = base_url + query
    response = requests.get(full_url)
    soup = BeautifulSoup(response.content, 'lxml')
    articles = [entry.text for entry in soup.find_all('span', class_='title')]

    return articles[:10]


# 定义 App 类，它继承自 QMainWindow 和 Ui_MainWindow。QMainWindow 是 PyQt5 中的主窗口类，而 Ui_MainWindow 是我们自定义的用户界面。
class App(QMainWindow, Ui_MainWindow):
    # 初始化函数
    def __init__(self):
        # 调用父类的初始化函数，确保 QMainWindow 和 Ui_MainWindow 的初始化也被执行
        super().__init__()

        # 调用 setupUi 方法（这是 Ui_MainWindow 中的方法）来设置和初始化界面元素
        self.setupUi(self)

        # 绑定按钮事件：当 pushButton 被点击时，执行 self.fetch_and_display 函数
        self.pushButton.clicked.connect(self.fetch_and_display)

    # 定义 fetch_and_display 函数
    def fetch_and_display(self):
        # 从 lineEdit 控件获取文本（作者名）
        author_name = self.lineEdit.text()

        try:
            # 尝试调用 fetch_articles_by_author 函数（该函数未在此代码中定义）来获取作者的文章
            articles = fetch_articles_by_author(author_name)
            # 在 textEdit 控件中显示文章，文章之间用换行符分隔
            self.textEdit.setText('\n'.join(articles))
        except Exception as e:
            # 如果在上述过程中出现任何错误，显示一个临界错误消息框
            QMessageBox.critical(self, "Error", str(e))


# 如果此脚本是直接运行，而不是作为模块导入，则执行以下代码
if __name__ == '__main__':
    # 创建一个 QApplication 对象，这是每个 PyQt5 应用程序都需要的
    app = QApplication(sys.argv)

    # 创建 App 类的实例
    window = App()

    # 显示主窗口
    window.show()

    # 开始应用程序的事件循环，并在窗口关闭时退出
    sys.exit(app.exec_())
