import os
import face_recognition
import cv2

# 读取到数据库中的人名和面部特征
# 1.准备工作
face_databases_dir='F:\\deep learning\\camera\\face_databases'
user_names=[] #存用户姓名
user_faces_encodings=[] #存用户面部特征向量

#2. 正式工作 
#2.1 得到face_databases_dir文件夹下所有的文件名
files=os.listdir('F:\\deep learning\\camera\\face_databases')
# 2.2 循环读取文件名进行进一步的处理
for image_shot_name in files:
    #2.2.1 截取文件名的前面那部分作为用户名存入user_names的列表中
    user_name,_=os.path.splitext(image_shot_name)
    user_names.append(user_name)

    #2.2.2 读取图片文件中的面部特征信息存入user_faces_encoding列表中
    image_file_name=os.path.join(face_databases_dir,image_shot_name)
    image_file=face_recognition.load_image_file(image_file_name)
    face_encoding=face_recognition.face_encodings(image_file)[0]
    

    user_faces_encodings.append(face_encoding)

# 打开摄像头，读取摄像头拍摄画面
# 定位到画面中人的脸部，并用绿色的框框把人的脸部框柱
# 用拍摄到人的脸部特征和数据库中的面部特征去匹配
# 并在用户头像的绿框上方用用户的姓名做标识，未知用户统一使用Unknown



#1.打开摄像头，获取摄像头对象
video_capture=cv2.VideoCapture(0)

# 2. 循环不停的去获取摄像头拍摄到的画面，并做进一步处理
while True:

    #2.1 获取拍到的画面
    ret,frame=video_capture.read()
    #2.2 从拍摄到的画面中提取出人脸所在区域(可能有多个)
    face_locations=face_recognition.face_locations(frame)
    face_encodings=face_recognition.face_encodings(frame,face_locations)

    #2.22 定义用于存储拍摄到的用户的姓名的列表
    # 如果特征匹配不上数据库中的特征，则是Unknown

    names=[]
    # 遍历face_encodings 和之前数据库中面部特征做匹配
    # 面部特征和数据库
    #并在用户头像的绿框上方用用户的姓名做标识，未知Unknown
    for face_encoding in face_encodings:
        matchs=face_recognition.compare_faces(user_faces_encodings,face_encoding)

        name="Unknow"
        for index,is_match in enumerate(matchs):
            if is_match:
                name=user_names[index]
                break
        names.append(name)

    #2.3循环遍历人的脸部所在区域，并画框，框上写名字
    #zip(['第1个人'，'第2个人'])




    for top,right,bottom,left in face_locations:
        # 2.3.1在人像所在区域画框
        color=(0,255,0)
        if names in user_names:
            color=(0,0,255)

            
        cv2.rectangle(frame,(left,top),(right,bottom),color,2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left,top-10),font,0.5,color,1)


    #2.4通过opencv把拍摄到的并画了框的画面展示出来
    cv2.imshow("Video",frame)
    #2.5 设定按q退出While循环，退出程序的这样一个机制
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#3. 退出程序的时候，释放摄像头或其他资源
video_capture.release()
cv2.destoryAllWindows()