import pandas
from matplotlib import pyplot as plt
import scipy.io
import datetime
mat = scipy.io.loadmat(r"C:\Users\haiha\Downloads\wiki.tar\wiki\wiki.mat")


d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
# dtype=[('dob', 'O'), ('photo_taken', 'O'), ('full_path', 'O'), ('gender', 'O'), ('name', 'O'), ('face_location', 'O'), ('face_score', 'O'), ('second_face_score', 'O')])
data = pandas.read_csv(r"C:\Work\repos\Delta_Age_AdaIN\datasets\megaage_asian\megaage_asian.csv")
print("a")

plt.subplot(1, 2, 1)
plt.hist(data['age'].to_numpy()[data['type'].to_numpy() == "training"], bins=50)
plt.title("Training")
plt.grid()
plt.subplot(1, 2, 2)
plt.hist(data['age'].to_numpy()[data['type'].to_numpy() == "testing"], bins=50)
plt.title("Test")
plt.grid()
plt.show()
