# 上传文件至github
    
    cd C:/Users/WHR/ustc-whr.github.io
    
    mkdir DL_PJ_1/TEST5
    
    cp D:/whr_laptop/2023Summer/TEST/4thGrade_a/DL_pragmatic/DL_PJ1/README_SUMMARY/SUMMARY.md DL_PJ_1/TEST5
    cp D:/whr_laptop/2023Summer/TEST/4thGrade_a/DL_pragmatic/DL_PJ1/TEST4/20231017_stock.py D:/whr_laptop/2023Summer/TEST/4thGrade_a/DL_pragmatic/DL_PJ1/TEST4/main_window_2.py D:/whr_laptop/2023Summer/TEST/4thGrade_a/DL_pragmatic/DL_PJ1/TEST4/STOCK_APP.png DL_PJ_1/TEST4
    
    git add DL_PJ_1/TEST5
    git add DL_PJ_1/TEST4
    
    git commit -m"Added to DL_PJ_1/TEST5 folder"
    git commit -m"Added to DL_PJ_1/TEST4 folder"
    
    git push origin main

# 将.ui文件转化为.py文件

    pyuic5 -o main_window_2.py main_window_2.ui

#  爬虫爬取数据

    import requests
    from bs4 import BeautifulSoup
    
    url='your_url'
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

    soup=BeautifulSoup(response.text,'html.parser')# html.parser解析器，速度快，容错能力强，但只支持HTML
    或者soup=BeautifulSoup(response.text,'lxml')# lxml解析器速度快，容错能力强，支持HTML和XML，但需要安装C语言库
    或者soup=BeautifulSoup(response.text,'html5lib')# html5lib解析器以浏览器的方式解析文档，生成HTML5格式的文档，但速度慢，容错能力强
    或者soup=BeautifulSoup(response.text,'xml')# xml解析器速度快，唯一支持XML的解析器
    或者soup=BeautifulSoup(response.content,'html.parser')# 用response.content解析网页内容,不需要指定编码，但是会丢失编码信息，可能会乱码

    selector = 'your_selector'
    target_data=soup.select_one(selector)
    target_data.text.strip()# 去除空格和换行符

# 建立函数和ui界面的联系

步骤如下：
    
    1.用qt designer设计界面：选取mian_window意味着这个界面是主界面，然后在右侧的object inspector中选取widget，然后在左侧的property editor中选取object name
    
    2.将.ui文件转化为.py文件
    
    3.直接调用.py文件中的object name即可
    
    4.建立class，将object name作为参数传入class中，然后在class中定义函数，最后在主函数中调用class中的函数即可
    
    5.如果有button,则需要在class中定义一个槽函数，然后将button的clicked信号与槽函数相连，这样当button被点击时，槽函数就会被调用
    
    eg:self.pushButton_2.clicked.connect(self.save_to_csv)
    
    6.运用try,exception语句，当try中的语句出现错误时，就会执行except中的语句

7.主函数：

if __name__=='__main__':

    app=QApplication(sys.argv)#from PyQt5.QtWidgets import QApplication

    myshow=APP()#自己定义的class

    myshow.show()#显示界面

    sys.exit(app.exec_())#app.exec_()表示程序一直执行，直到主窗口关闭，sys.exit()方法确保主循环安全退出

