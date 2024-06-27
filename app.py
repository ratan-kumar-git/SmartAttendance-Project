from flask import Flask, render_template, request, redirect, session, flash, Response,url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func
from sklearn.neighbors import KNeighborsClassifier
import os
import bcrypt
import cv2
import numpy as np
import datetime
import pickle


app = Flask(__name__)


# Make Connection with Data Base
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'Bablu@12345'


# Create Database with the name of "User" for Orgnization or School
class User(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = self.hash_password(password)

    def hash_password(self, password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


# Create Database with the name of "Student" for save data of Orgnization or School
class Student(db.Model): 
    st_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    st_name = db.Column(db.String(100), nullable=False)
    roll_no = db.Column(db.String(100), unique=True, nullable=False)
    branch = db.Column(db.String(100), nullable=False)
    year = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    user = db.relationship('User', backref=db.backref('students', lazy=True))

    def __init__(self, user_id, st_name, roll_no, branch, year):
        self.user_id = user_id
        self.st_name = st_name
        self.roll_no = roll_no
        self.branch = branch
        self.year = year

class Attendance(db.Model):
    atten_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # student_id = db.Column(db.Integer, db.ForeignKey('student.st_id'), nullable=False)
    st_name = db.Column(db.String(100), nullable=False)  # Added student name attribute
    roll_no = db.Column(db.String(100), nullable=False)  # Added roll_no attribute
    branch = db.Column(db.String(100), nullable=False)  # Added branch attribute
    year = db.Column(db.String(100), nullable=False)  # Added year attribute
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    

    user = db.relationship('User', backref=db.backref('attendances', lazy=True))
    # student = db.relationship('Student', backref=db.backref('attendances', lazy=True))

    def __init__(self, user_id, st_name, roll_no, branch, year, date, time):
        self.user_id = user_id
        self.st_name = st_name
        self.roll_no = roll_no
        self.branch = branch
        self.year = year
        self.date = date
        self.time = time
        
        


# Create Database with the name of "User"
class Contactus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.String(200))

    def __init__(self, name, email, message):
        self.name = name
        self.email = email
        self.message = message

with app.app_context():
    db.create_all()



            ######################################
            ### All Functions are written Here ###
            ######################################


capture = False
captured_student_name = ""
captured_roll_no = ""

def add_student_face():
    camera = cv2.VideoCapture(cv2.CAP_ANY)
    global capture, captured_student_name, captured_roll_no
    face_data = []
    i = 0

    # Create the directory for storing data if it doesn't exist
    if not os.path.exists('static/data'):
        os.makedirs('static/data')

    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if capture:
        while True:
            ret, frame = camera.read()
            if ret: 
                flipped_frame = cv2.flip(frame, 1)
                # Convert the frame to grayscale
                gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)

                for (a, b, w, h) in face_coordinates:
                    # Draw a rectangle around the face
                    cv2.rectangle(flipped_frame, (a, b), (a+w, b+h), (255, 0, 0), 2)

                    # Extract the face region
                    faces = flipped_frame[b:b+h, a:a+w, :]

                    # Resize the face region to 50x50 pixels
                    resized_faces = cv2.resize(faces, (50, 50))

                    # Append every 10th face to the face_data list
                    if i % 10 == 0 and len(face_data) < 10:
                        face_data.append(resized_faces)
                    
                    # Increment the frame count
                    i += 1
                    # Display the frame count on the frame
                    cv2.putText(flipped_frame, str(i), (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        

                # Break the loop if 'Esc' key is pressed or we have collected 10 face images
                if len(face_data) >= 10:
                    break  

                try:
                    ret, buffer = cv2.imencode('.jpg', flipped_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Error in encoding frame: {e}")

            else:
                print('Failed to capture frame')
                break
        capture = False  
        camera.release()


    # Convert face_data to a numpy array and reshape it
    face_data = np.asarray(face_data)
    face_data = face_data.reshape(10, -1)

    # Handle the case where face_data might be an empty array
    if face_data.size == 0:
        print("No face data collected.")
        return

    # Save the names and face data
    if 'names.pkl' not in os.listdir('static/data/'):
        names = [captured_student_name + "_" + captured_roll_no] * 10
        with open('static/data/names.pkl', 'wb') as file:
            pickle.dump(names, file)
    else:
        with open('static/data/names.pkl', 'rb') as file:
            names = pickle.load(file)
        names += [captured_student_name + "_" + captured_roll_no] * 10
        with open('static/data/names.pkl', 'wb') as file:
            pickle.dump(names, file)

    if 'faces.pkl' not in os.listdir('static/data/'):
        faces = face_data
        with open('static/data/faces.pkl', 'wb') as w:
            pickle.dump(faces, w)
    else:
        with open('static/data/faces.pkl', 'rb') as w:
            faces = pickle.load(w)
        if faces.size == 0:  # Handle the case where faces might be an empty array
            faces = face_data
        else:
            faces = np.append(faces, face_data, axis=0)
        with open('static/data/faces.pkl', 'wb') as w:
            pickle.dump(faces, w)



def take_atten_face_reco():
    camera = cv2.VideoCapture(0,cv2.CAP_ANY)
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    with open('static/data/faces.pkl', 'rb') as w:
        faces = pickle.load(w)

    with open('static/data/names.pkl', 'rb') as file:
        labels = pickle.load(file)

    print('Shape of Faces matrix --> ', faces.shape)
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(faces, labels)


    while True:
        ret, frame = camera.read()
        if ret:
            flipped_frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)

            for (a, b, w, h) in face_coordinates:
                fc = flipped_frame[b:b + h, a:a + w, :]
                r = cv2.resize(fc, (50, 50)).flatten().reshape(1,-1)
                text = knn.predict(r)

                # Convert NumPy array element to string
                text = text[0]

                # Split the string based on "_"
                split_text = text.split("_")

                name = split_text[0]
                roll_no = split_text[1]

                cv2.rectangle(flipped_frame, (a, b), (a + w, b + w), (0, 0, 255), 2)

                with app.app_context():
                    # Find student exists in the database
                    existing_student = Student.query.filter_by(st_name=name, roll_no=roll_no).first()
                    if existing_student:
                        user_id = existing_student.user_id
                        branch = existing_student.branch
                        year = existing_student.year
                        current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
                        current_date = datetime.datetime.now().strftime("%d-%b-%Y")
                        
                        # Check if the student is already present in the database for the current date
                        check_today_atten = Attendance.query.filter_by(st_name=name, roll_no=roll_no, date=current_date).first()
                        if check_today_atten:
                            text = f"{name} your attendance marked"
                            cv2.putText(flipped_frame, text, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                           
                        else:
                            cv2.putText(flipped_frame, text, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            new_student_atten = Attendance(user_id=user_id, st_name=name, roll_no=roll_no, branch=branch, year=year, date=current_date, time=current_time)
                            db.session.add(new_student_atten)
                            db.session.commit()
                            # updated_atte = Attendance.query.filter_by(user_id = user_id, date=current_date).all()



            try:
                ret, buffer = cv2.imencode('.jpg', flipped_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
        else:
            print("Failed to capture frame")
            break
    camera.release()



            ######################################
            ### Here All Endpoints are written ###
            ######################################

# Home Page 
@app.route('/')
def index():
    return render_template('home.html', active_page='home', title='Home | SmartAttendance')


# Register Page 
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already exists. Please use different email.', 'error')
            return redirect('/register')  # Redirect back to registration page with flash message

        
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect('/register')

    return render_template('register.html', title='Signup | SmartAttendance')


# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            error = 'Invalid email or password. Please try again.'
            flash(error, 'error')

    return render_template('login.html', title='Login | SmartAttendance')


# Dashboard Page
@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        current_date = datetime.datetime.now().strftime("%d-%b-%Y")
        today_atten = Attendance.query.filter_by(user_id = user.id, date=current_date).all()
        return render_template('dashboard.html', active_page='dashboard', user=user, title='Dashboard', today_atten=today_atten)
    else:
        flash('You need to login first.', 'error')
        return redirect('/login')


# Dashboard Add User Page
@app.route('/add_user', methods=['GET', 'POST'])
def dashboard_add_user():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        students = Student.query.filter_by(user_id=user.id).all()
        global capture, captured_student_name, captured_roll_no
        if request.method == 'POST':
            st_name = request.form['st_name']
            roll_no = request.form['roll_no']
            branch = request.form['branch']
            year = request.form['year']
           
            existing_roll_no = Student.query.filter_by(roll_no=roll_no).first()
            if existing_roll_no:
                flash('Roll No already exists.', 'error')
                return redirect('/add_user')
            else:
                 # Capture photo from webcam
                if request.form.get('click') == 'Register & Capture Photo':
                    capture = True
                    captured_student_name = st_name
                    captured_roll_no = roll_no
                # flash('all the input array dimensions except for the concatenation axis must match exactly', 'error')
            new_student = Student(user_id=user.id, st_name=st_name, roll_no=roll_no, branch=branch, year=year)
            db.session.add(new_student)
            db.session.commit()
            students = Student.query.filter_by(user_id=user.id).all()
            flash('Photo capturing Start   ==>   wait till photo Count is 90   ==>   Student added successfully', 'success')
        return render_template('add_user.html', active_page='dashboard', user=user, title='Dashboard | Add Student', students=students)
    else:
        flash('You need to login first.', 'error')
        return redirect('/login')


@app.route('/add_student')
def add_student_video_feed():
    return Response(add_student_face(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Dashboard Edit User Page
@app.route('/edit_user')
def dashboard_edit_user():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        students = Student.query.filter_by(user_id=user.id).all()
        return render_template('edit_user.html', active_page='dashboard', user=user, title='Dashboard | All Students', students=students)
    else:
        flash('You need to login first.', 'error')
        return redirect('/login')


#Edit student data    
@app.route('/edit_user/<int:student_id>', methods=['GET', 'POST'])
def edit_user(student_id):
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        student = Student.query.filter_by(st_id=student_id, user_id=user.id).first()

        if not student:
            flash('Student not found or you do not have permission to edit this student.', 'error')
            return redirect('/edit_user')

        if request.method == 'POST':
            if student:
                student.st_name = request.form['st_name']
                student.roll_no = request.form['roll_no']
                student.branch = request.form['branch']
                student.year = request.form['year']

                try:
                    db.session.commit()  # Commit the transaction to save the changes
                    flash('Student data updated successfully.', 'success')
                    return redirect('/edit_user')
                except IntegrityError:
                    db.session.rollback()  # Rollback the transaction in case of integrity error
                    flash('This roll number already exists.', 'error')
                    return redirect(url_for('edit_user', student_id=student_id))
                except Exception as e:
                    db.session.rollback()  # Rollback the transaction in case of other errors
                    flash(f'An error occurred while updating the student data: {e}', 'error')
                    return redirect(url_for('edit_user', student_id=student_id))
        
        return render_template('edit_user1.html', active_page='dashboard', user=user, student=student, title='Dashboard | Edit Student')
    else:
        flash('You need to login first.', 'error')
        return redirect('/login')


@app.route('/delete_user/<int:student_id>', methods=['GET', 'POST'])
def delete_user(student_id):
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        student = Student.query.filter_by(st_id=student_id, user_id=user.id).first()

        if student:
            db.session.delete(student)
            db.session.commit()
            flash('Student deleted successfully.', 'success')
        else:
            flash('Student not found or you do not have permission to delete this student.', 'error')

        return redirect('/edit_user')
    else:
        flash('You need to login first.', 'error')
        return redirect('/login')


# Dashboard Take Attendance Page
@app.route('/take_attendance', methods=['GET', 'POST'])
def dashboard_take_attendance():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        current_date = datetime.datetime.now().strftime("%d-%b-%Y")
        today_atten = Attendance.query.filter_by(user_id = user.id, date=current_date).all()
        return render_template('take_attendance.html', active_page='dashboard', user=user, today_atten=today_atten, title='Dashboard | Take Attendance')
    else:
        flash('You need to login first.', 'error')
        return redirect('/login')


@app.route('/take_atten')
def take_atte_video_feed():
    return Response(take_atten_face_reco(), mimetype='multipart/x-mixed-replace; boundary=frame')  


# Dashboard Privious Attendance Page
@app.route('/pre_attendance', methods=['GET', 'POST'])
def dashboard_pre_attendance():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        today_date = datetime.datetime.now().strftime('%Y-%m-%d') # in YY/MM/DD Formate
        date = datetime.datetime.now().strftime("%d-%b-%Y") # in DD/MM/YY Formate
        today_atten = Attendance.query.filter_by(user_id = user.id, date=date).all()

        if request.method == 'POST':
            date = request.form['date']
            today_date = date
            date = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%d-%b-%Y")
            today_atten = Attendance.query.filter_by(user_id=user.id, date=date).all()   
        return render_template('pre_attendance.html', active_page='dashboard', user=user, today_atten=today_atten,  title='Dashboard | Privious Attendance', today_date=today_date)
    else:
        flash('You need to login first.', 'error')
        return redirect('/login')


#Logout Page
@app.route('/logout')
def logout():
    session.pop('email', None)
    flash('You have been logged out.', 'info')
    return redirect('/login')


#ContactUs Page
@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        new_contact = Contactus(name=name, email=email, message=message)
        db.session.add(new_contact)
        db.session.commit()
        flash('Message Sent Successfully.', 'success')
        return redirect('/contactus')

    return render_template('contactus.html', active_page='contactus', title='Contact Us')


#About Page
@app.route('/about')
def about():
    return render_template('about.html', active_page='about', title='About')


if __name__ == '__main__':
    app.run(debug=True)
