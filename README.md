# 万物检测 on 1684X
## 功能描述
基于RAM实现万物检测，可以高精度地识别任何常见类别，当与定位模型（Grounded-SAM）结合使用时，RAM形成了一个强大的通用视觉语义分析管道。
优势总结如下：
- 具有出色的图像标记功能，具有强大的零样本泛化功能;
- 开源;
- 具有显著的灵活性，可满足各种应用场景的需求。

## 安装
1. 拉取代码仓库，`git clone https://github.com/ZillaRU/AnnoAnything.git`，并`cd AnnoAnything`进入项目根目录。
2. 安装依赖库，`pip3 install -r requirements.txt`
3. 替换tpu_perf安装目录下的infer.py文件, 默认安装在"/home/$USER/.local/lib/python3.*/site-packages/tpu_perf/", 请根据具体环境确定路径。
    ```bash
    cp ./replace-this-file/infer.py /home/linaro/.local/lib/python3.8/site-packages/tpu_perf/ 
    ```
    若用了virtualenv，则被换掉的文件不是该路径，可`python3 -c "import tpu_perf; print(tpu_perf.__file__)"`查看。
4. 下载bmodel，并解压（`unzip annoanything-bmodel.zip`）放在该项目根目录的`bmodel`文件夹。
   - sftp下载：
       ```bash
       pip3 install dfss --upgrade
       python3 -m dfss --url=open@sophgo.com:/aigc/annoanything-bmodel.zip
       ```
   - 或 [Google drive](https://drive.google.com/drive/folders/1WFfq32nKCYhEwJvCZV5XYw9sFRriZYqP?usp=sharing)
5. 运行demo，`python3 app.py`。等待模型加载完毕，终端会提示端口号，通过浏览器用`本机ip:端口号`即可访问。
    - 示例：检测图中所有物体，输出tag并标出位置
      <img width="848" alt="示例：输出tag并标出位置" src="https://github.com/ZillaRU/AnnoAnything/assets/25343084/6a77ed66-3555-48c4-a58f-d3d52b2290fa">
    - 示例：描述需要检测的物体，并检测
      <img width="848" alt="示例：输出tag并标出位置" src="https://github.com/ZillaRU/AnnoAnything/assets/25343084/7248472e-b0e3-46b6-bd1b-c5b5df13a0d5">

    

