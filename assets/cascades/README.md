将 `haarcascade_frontalcatface.xml` 放到这个目录下。

检测模块会按以下顺序寻找级联文件：

1. `configs/*.yaml` 中显式指定的 `cascade_path`
2. `assets/cascades/haarcascade_frontalcatface.xml`
3. OpenCV 安装目录中的 `haarcascade_frontalcatface*.xml`
