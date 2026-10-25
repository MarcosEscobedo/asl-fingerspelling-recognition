\# Real-Time ASL Fingerspelling Recognition



A computer vision and machine learning project for real-time recognition of static American Sign Language (ASL) fingerspelling using hand landmarks extracted from a webcam.



The system uses \*\*MediaPipe\*\* to track 21 hand landmarks and a \*\*Random Forest classifier\*\* to recognize 24 static fingerspelling letters in real time.



\## Project Overview



The objective of this project is to explore how hand pose information can be used to recognize ASL fingerspelling through computer vision and machine learning.



The current system recognizes the following 24 static letters:



\*\*A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y\*\*



The letters \*\*J and Z are not included\*\* because they depend on hand motion and therefore require temporal information rather than a single-frame hand pose.



\## System Pipeline



```text

Webcam

&#x20;  ↓

OpenCV

&#x20;  ↓

MediaPipe Hand Landmarker

&#x20;  ↓

21 hand landmarks

&#x20;  ↓

Landmark normalization

&#x20;  ↓

Random Forest classifier

&#x20;  ↓

ASL fingerspelling prediction

&#x20;  ↓

Temporal prediction stabilization

```



\## Technologies



\* Python

\* OpenCV

\* MediaPipe

\* pandas

\* scikit-learn

\* joblib



\## Dataset



A custom dataset was collected using a webcam and MediaPipe Hand Landmarker.



Each sample contains:



\* ASL letter label

\* Participant ID

\* Session ID

\* Handedness

\* 21 hand landmarks

\* X, Y and Z coordinates for every landmark



Each hand therefore contributes \*\*63 numerical landmark features\*\*.



\### Data Collection



The dataset contains:



| Session   | Samples per letter | Letters |     Total |

| --------- | -----------------: | ------: | --------: |

| P01-S01   |                 30 |      24 |       720 |

| P01-S02   |                 15 |      24 |       360 |

| \*\*Total\*\* |             \*\*45\*\* |  \*\*24\*\* | \*\*1,080\*\* |



Two separate recording sessions were used to introduce variation in hand position, distance, orientation and capture conditions.



Landmark Normalization



Raw landmark coordinates depend on the position and apparent size of the hand in the image.



To reduce this dependency:



The wrist landmark is used as the origin.

Wrist coordinates are subtracted from all landmarks.

Coordinates are scaled relative to the largest absolute coordinate value.



This produces a representation focused primarily on hand geometry rather than image position.



Machine Learning Model



A Random Forest classifier was trained using the normalized hand landmarks.



The initial proof of concept used six visually distinct letters:



A, B, C, L, V and Y



A random train/test split produced 100% accuracy. However, samples collected consecutively during the same recording session are highly similar, making this evaluation overly optimistic.



For this reason, the final evaluation was redesigned as a cross-session evaluation.



Cross-Session Evaluation



The classifier was trained exclusively on:



P01-S01 → 720 samples



and evaluated exclusively on:



P01-S02 → 360 unseen session samples

Result



Cross-session accuracy: 86.67%



The model correctly classified:



312 / 360 samples



with 48 classification errors.



This evaluation provides a more realistic estimate of performance than randomly mixing samples from the same recording session.



Error Analysis



Performance varied substantially between letters.



Several letters achieved perfect or near-perfect recognition, including:



B

D

E

F

G

H

K

P

Q

W

X



The main confusion groups were:



M / N / T



These letters have similar hand configurations and depend strongly on subtle differences in thumb placement.



Examples observed during evaluation included:



T → N

N → M

M → N

U / V / R



These configurations rely heavily on the relationship between the index and middle fingers.



Common errors included:



V → U

U → V

R → U

R → V

I / Y



Several instances of I were classified as Y, suggesting that thumb position is another important discriminative feature.



These errors indicate that hand-landmark geometry alone may not always capture subtle distinctions robustly across different orientations.



Real-Time Recognition



The final classifier was trained using all 1,080 available samples from S01 and S02 and integrated into a webcam-based recognition application.



Predictions are stabilized using a short temporal voting window.



Rather than displaying every frame-level prediction independently, the system examines the seven most recent predictions and displays a letter when at least five agree.



This reduces prediction flickering between visually similar classes.



Linguistic Extension Using ASL-LEX



To explore how the system could eventually expand beyond fingerspelling, ASL-LEX 2.0 was included as a small linguistic analysis component.



The dataset contains information for 2,723 lexical signs and includes variables such as:



subjective sign frequency

lexical class

iconicity

phonological properties

dominant translation



A Python script was used to identify lexical signs with high subjective-frequency ratings.



Examples among the highest-rated signs included:



water

you

I\_love\_you

yes

eat

want

me

work

book

mother

sleep



These values represent subjective frequency ratings, not direct counts of sign occurrences in natural conversations.



The analysis provides a linguistically informed way of selecting vocabulary for future expansion of the recognition system.



Limitations



The current system has several important limitations.



Static recognition



Only static fingerspelling configurations are recognized.



J and Z require movement, so they cannot be represented adequately by a single set of hand landmarks.



Single participant



The current custom dataset was collected from one participant.



The 86.67% evaluation therefore measures cross-session generalization, not generalization to unseen users.



Hand landmarks only



The model only considers hand geometry.



A more complete ASL recognition system may require:



hand movement

both hands

body pose

facial expressions

non-manual markers

temporal context

Similar handshapes



Some fingerspelling letters have subtle geometric differences that are difficult to distinguish consistently using the current representation.



Future Work



Possible extensions include:



Collect data from multiple participants.

Evaluate performance on completely unseen users.

Add temporal modeling for J and Z.

Investigate trajectory-based features or recurrent/sequence models.

Expand from fingerspelling to lexical ASL signs.

Use ASL-LEX frequency information to guide vocabulary selection.

Incorporate pose and facial information for richer sign-language recognition.

Compare Random Forest performance against neural-network approaches.

Project Structure

asl-visual-coach/

│

├── data/

│   ├── asl\_static.csv

│   ├── pilot\_landmarks.csv

│   ├── signdata.csv

│   └── asllex\_top15.csv

│

├── models/

│   ├── hand\_landmarker.task

│   ├── asl\_classifier\_s01.joblib

│   └── asl\_classifier\_final.joblib

│

├── analyze\_asllex.py

├── camera\_test.py

├── capture\_data.py

├── capture\_dataset.py

├── evaluate\_sessions.py

├── hand\_tracking.py

├── realtime\_final.py

├── realtime\_recognition.py

├── train\_final\_model.py

├── train\_model.py

├── requirements.txt

└── README.md

Running the Project



Activate the Python environment and install dependencies:



pip install -r requirements.txt



Run the final real-time recognition system:



python realtime\_final.py



Press Q to close the webcam window.



Key Result



A Random Forest classifier using normalized MediaPipe hand landmarks achieved 86.67% cross-session accuracy across 24 static ASL fingerspelling classes.



The project demonstrates an end-to-end workflow involving custom dataset creation, computer vision, machine learning, experimental evaluation, real-time inference, error analysis and a small linguistically informed extension using ASL-LEX.


## Data Attribution

The lexical analysis component uses data from **ASL-LEX 2.0**, a lexical and phonological database containing information for 2,723 American Sign Language signs.

ASL-LEX data, excluding the sign reference videos, are distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

References:

Sevcikova Sehyr, Z., Caselli, N., Cohen-Goldberg, A. M., & Emmorey, K. (2021). The ASL-LEX 2.0 Project: A database of lexical and phonological properties for 2,723 signs in American Sign Language. *Journal of Deaf Studies and Deaf Education*.

Caselli, N., Sevcikova Sehyr, Z., Cohen-Goldberg, A. M., & Emmorey, K. (2017). ASL-LEX: A lexical database of American Sign Language. *Behavior Research Methods, 49*(2), 784–801.




