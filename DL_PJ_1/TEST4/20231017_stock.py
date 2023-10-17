from bs4 import BeautifulSoup
import requests
import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from main_window_2 import Ui_MainWindow
import os
import random



# %%
def fetch_table_data(CODE, name):
    url = f"https://www.macrotrends.net/stocks/charts/{CODE}/{name}/stock-price-history"
    headers_list = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'},
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'},
        {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'},
        {
            'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'},
        {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Android 10; Mobile; rv:89.0) Gecko/89.0 Firefox/89.0'}
    ]

    selected_headers = random.choice(headers_list)
    response = requests.get(url, headers=selected_headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # 使用选择器来定位表格
        table = soup.select_one('#main_content > div:nth-child(8) > table')

        # 如果找到了表格，解析数据
        if table:
            # 用pandas.DataFrame来保存表格
            df = pd.read_html(str(table))[0]
            df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Close', 'Change']
            return df


        else:
            print("Table not found!")
            return None
    elif response.status_code == 403:
        print('Access denied!')
        return None
    else:
        print(f"Error {response.status_code}: Unable to fetch page")
        return None

def fetch_table_data_alternative(CODE,name,mission):
    mission=int(mission)
    index=['total-assets','revenue','pe-ratio'][mission-1]
    url=f'https://www.macrotrends.net/stocks/delisted/{CODE}/{name}/{index}'
    headers_list = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'},
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'},
        {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'},
        {
            'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'},
        {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Android 10; Mobile; rv:89.0) Gecko/89.0 Firefox/89.0'}
    ]

    selected_headers = random.choice(headers_list)
    response = requests.get(url, headers=selected_headers)

    soup = BeautifulSoup(response.content, 'html.parser')
    if mission in [1,2]:
        table1 = soup.select_one('#style-1 > div:nth-child(1) > table')
        table2 = soup.select_one('#style-1 > div:nth-child(2) > table')
        df1 = pd.read_html(str(table1))[0]
        df2 = pd.read_html(str(table2))[0]
        if mission==1:
            df1.columns=['Year','Assets(m)']
            df2.columns=['Y_Quarter','Assets(m)']
        else:
            df1.columns=['Year','Revenue(m)']
            df2.columns=['Y_Quarter','Revenue(m)']
        df = pd.concat([df1, df2],axis=1)#, left_index=True, right_index=True)
        return df
    elif mission==3:
        table = soup.select_one('#style-1 > table')
        df = pd.read_html(str(table))[0]
        df.columns=['Date','price','TTM_NET_EPS','PE_Ratio']
        return df
    else:
        return ['MISSION INDEX ERROR']




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

        self.pushButton_2.clicked.connect(self.save_to_csv)

        # 初始化 df
        self.df = None
        self.df1 = None
        self.df2 = None
        self.name = None
        self.CODE = None

    def save_to_csv(self):
        # 检查 df 是否有内容
        if self.df is not None:
            # 获取桌面路径
            desktop = os.path.expanduser("~/Desktop")
            # 定义默认的文件名
            default_filename = f"{self.CODE}_{self.name}_data.csv"
            # 获取完整的文件路径
            full_path = os.path.join(desktop, default_filename)

            # 保存 DataFrame 到 .csv 文件
            self.df.to_csv(full_path, index=False)
            QMessageBox.information(self, "Success", f"Data saved to {full_path}")
        else:
            QMessageBox.warning(self, "Warning", "No data available to save!")


    # 定义 fetch_and_display 函数
    def fetch_and_display(self):
        # 从 lineEdit 控件获取文本（作者名）
        CODE = self.lineEdit.text()
        name = self.lineEdit_2.text()
        mission= self.lineEdit_3.text()
        self.name=name
        self.CODE=CODE

        try:
            # 尝试调用 fetch_articles_by_author 函数（该函数未在此代码中定义）来获取作者的文章
            #df1 = fetch_table_data(CODE, name)
            df1= fetch_table_data('TIF', 'tiffany')
            df1=fetch_table_data('LFCHY','china-life-insurance')
            self.df1=df1
            # 在 textEdit 控件中显示df,df是一个DataFrame,要美观地显示，需要转换为字符串
            self.textEdit.setText(df1.to_string())
            #df2= fetch_table_data_alternative(CODE,name,mission)
            df2= fetch_table_data_alternative('TIF', 'tiffany','1')
            df2=fetch_table_data_alternative('LFCHY','china-life-insurance','1')
            self.df2=df2
            self.textEdit_2.setText(df2.to_string())
            self.df=pd.concat([df1,df2],axis=1)

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
