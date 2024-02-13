import torch

# course_l_x, course_l_yの仮定した値をPyTorchテンソルに変換
course_l_x = torch.tensor([1, 2, 3, 4, 5])
course_l_y = torch.tensor([10, 20, 30, 40, 50])

# course_l_xとcourse_l_yを結合して新しいテンソルcourse_lを作成
print(course_l_x.size())
course_l = torch.stack((course_l_x, course_l_y), dim=1)
print(course_l.size())
# course_lからcourse_l_x_revisedを取得
course_l_x_revised = course_l[:, 0]
course_l_y_revised = course_l[:, 1]
print("course_l_x_revised:", course_l_x_revised)
print("course_l_x_revised:", course_l_y_revised)
