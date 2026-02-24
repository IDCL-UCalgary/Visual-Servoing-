from ultralytics import YOLO
import matplotlib.pyplot as plt
import pickle
# Set global font and size
plt.rcParams.update({
    "font.size": 40,          # change font size
    "font.family": "Times New Roman"    # change font family
})

student_model = YOLO("/home/eeel126/Desktop/ENEL645Project/runs/detect/Student Training/weights/student_best.pt")
teacher_model = YOLO("/home/eeel126/Desktop/ENEL645Project/runs/detect/fold2/weights/best.pt")
pruned_model = YOLO("/home/eeel126/Desktop/ENEL645Project/Car_final.pt")

# Run validation
student_results = student_model.val(data="/home/eeel126/Desktop/ENEL645Project/overhead_vehicle_detection/data.yaml", plots=True)  # plots=True generates PR curves, etc.
teacher_results = teacher_model.val(data="/home/eeel126/Desktop/ENEL645Project/overhead_vehicle_detection/data.yaml", plots=True)  # plots=True generates PR curves, etc.
pruned_results = pruned_model.val(data="/home/eeel126/Desktop/ENEL645Project/overhead_vehicle_detection/data_pruned.yaml", plots=True)  # plots=True generates PR curves, etc.


# save results as pickle object
with open("student.pkl", "wb") as f:
    pickle.dump(student_results, f)
    
with open("teacher.pkl", "wb") as f:
    pickle.dump(teacher_results, f)

with open("pruned.pkl", "wb") as f:
    pickle.dump(pruned_results, f)

# fig = plt.gcf()
# ax = plt.gca()

# # Update font sizes manually
# ax.title.set_fontsize(40)
# ax.xaxis.label.set_fontsize(30)
# ax.yaxis.label.set_fontsize(30)
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(24)

# # Save PR curve manually if you want
# plt.savefig("PR_curve_custom_font.png", dpi=300)
# plt.show()

# 