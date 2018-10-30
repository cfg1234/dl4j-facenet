# dl4j-facenet
这是facenet的dl4j实现
环境要求：
1.JDK1.8或者更高版本
2.maven 3.3.9或者更高版本

编译：
1.在dl4j-facenet根目录下，运行mvn pacakge编译项目
2.运行./run.sh imgLibSample运行人脸识别测试代码(imgLibSample为人脸库，可以自行替换为自己的库，库的目录格式参考例子)
3.等待程序加载人脸库
4.看到提示"start detect  Insert image path:"的时候，输入要识别的图片所在路径(可以选择testImg中的图片进行测试)

注意：
1.人脸库和测试图片不要太大（jpg最好限制在100k以内），否则加载人脸库会很慢
2.检测图片中最好不要有多个人脸
