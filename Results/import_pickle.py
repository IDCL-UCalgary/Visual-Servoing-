import pickle
from matplotlib import pyplot as plt



import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,          # base font size
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})


# Load models
with open("student.pkl", "rb") as f:
    student = pickle.load(f)

with open("teacher.pkl", "rb") as f:
    teacher = pickle.load(f)

with open("pruned.pkl", "rb") as f:
    pruned = pickle.load(f)

def get_curves(obj):
    return {
        "rp": (obj.curves_results[0][0], obj.curves_results[0][1]),  # Recall vs Precision
        "cr": (obj.curves_results[1][0], obj.curves_results[1][1]),  # Confidence vs Recall
        "cp": (obj.curves_results[2][0], obj.curves_results[2][1]),  # Confidence vs Precision
    }

student_curves = get_curves(student)
teacher_curves = get_curves(teacher)
pruned_curves  = get_curves(pruned)

class_idx = 2  # Car detections

# Recall vs Precision
plt.figure()
plt.plot(student_curves["rp"][0], student_curves["rp"][1][class_idx, :], label="Student")
plt.plot(teacher_curves["rp"][0], teacher_curves["rp"][1][class_idx, :], label="Teacher")
plt.plot(pruned_curves["rp"][0],  pruned_curves["rp"][1][0, :],  label="Pruned")
plt.xlabel("Recall")
plt.ylabel("Precision")
#plt.title("Recall vs Precision")
plt.legend()
plt.grid(True)


# Confidence vs Recall
plt.figure()
plt.plot(student_curves["cr"][0], student_curves["cr"][1][class_idx, :], label="Student")
plt.plot(teacher_curves["cr"][0], teacher_curves["cr"][1][class_idx, :], label="Teacher")
plt.plot(pruned_curves["cr"][0],  pruned_curves["cr"][1][0, :],  label="Pruned")
plt.xlabel("Confidence")
plt.ylabel("Recall")
#plt.title("Confidence vs Recall")
plt.legend()
plt.grid(True)

# Confidence vs Precision
plt.figure()
plt.plot(student_curves["cp"][0], student_curves["cp"][1][class_idx, :], label="Student")
plt.plot(teacher_curves["cp"][0], teacher_curves["cp"][1][class_idx, :], label="Teacher")
plt.plot(pruned_curves["cp"][0],  pruned_curves["cp"][1][0, :],  label="Pruned")
plt.xlabel("Confidence")
plt.ylabel("Precision")
#plt.title("Confidence vs Precision")
plt.legend()
plt.grid(True)

plt.show()
