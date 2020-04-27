'''بسم الله الرحمن الرحيم'''

                            #Project 8 (Titanic Ship) using Real data set- sheet excel
import numpy as np
import pandas as pd
#مكتبة لاظهار العلاقات بين البيانات
import seaborn as sns
#مكتبة ال matploylib.pyplot هيا لعمل vizulaize ال data امام المستخدم بشكل مرئي يسطتيع فهمة
import matplotlib.pyplot as plt
%matplotlib inline
x_train=pd.read_csv("train.csv")
x_test=pd.read_csv("test.csv")
y_test=pd.read_csv("gender_submission.csv")
#column explain:
#1st: رقم الشيت
#2end(PassengerId): رقم الفرد على السفينة
#3rd(servived): حى او ميت 
#4th(Pclass): رتبة الكابينة
#5th(Name): اسم الفرد
#6th(sex): نوع الفرد
#7th(sibsp): سواء مراة او رجل - عدد الرجال لو مراة أو عدد النساء لو رجل اللذين معة على السفينة
#8th(Parch): عدد الاشخاص ذو علاقة دم بالشخص سواء ابن او اب او زوجة
#9th(Ticket): رقم التذكرة
#10(Fare): سعر التذكرة
#11th(Cabin): أسم الكابينة
#12th(Embarked): السفينة كانت مقسمة لطبقات المجتمع غنى فقير وسط..هنا هى طبقة الفرد فى السفينة
#######
#لاظهار اول 5 صفوف بالشيت (الاصل بدون ارقام بيظهر اول 5 صفوف)
x_train.head()
#لاظهار عدد صفوف معينة
x_train.head(10)
#هنقوم بمسح الاعمدة اللى مش عايزنها (اللى شايفين انها بالمنطق لاتؤثر على الاشخاص اللذين ماتوا او لا) مثل ترقيم ال ID للاشخاص و اسم الاشخاص و رقم التيكت و سعر التيكت و رقم الكابينة
# axis= 1 لتحديد انى عايز امسح العمود الفلانى من الاعمدة
x_test=x_test.drop(["PassengerId","Name","Ticket","Fare","Cabin"],axis=1)
x_train=x_train.drop(["PassengerId","Name","Ticket","Fare","Cabin"],axis=1)
x_test.head()
x_train.head()
#عشان فية بيانات غير معلومة =  Not A Number(NAN)d 
#عشان كدة لازم نفحص كل عمود هل فية قيم غير معرفة ولا لا
#الاول هنجمع كل القيم الغير معرفة عشان نعرف عددها فى كل عمود
x_train['Survived'].isna().sum()
#لا يوجد قيم مفقودة فى survived
x_train['Pclass'].isna().sum()
x_test['Pclass'].isna().sum()
#لا توجد قيم مفقودة فى ال test و ال train فى ال Pclass
x_train['Sex'].isna().sum()
x_test['Sex'].isna().sum()
#لا توجد قيم مفقودة بال train وال test فى عمود ال Sex
x_train['Age'].isna().sum()
x_test['Age'].isna().sum()
#يوجد 177 قيمة مفقودة بال train 
#وعدد 86 قيمة مفقودة بال test
x_train['SibSp'].isna().sum()
x_test['SibSp'].isna().sum()
#لا توجد قيم مفقودة بال train وال test فى عمود ال SibSp
x_train['Parch'].isna().sum()
x_test['Parch'].isna().sum()
#لا توجد قيم مفقودة بال train وال test فى عمود ال Parch
x_train['Embarked'].isna().sum()
x_test['Embarked'].isna().sum()
#يوجد 2 قيمة مفقودة بال train 
#ولا توجد قيم مفقودة بال test
#############
#دلوقتى عايزين نعوض القيم المفقودة فى الاعمدة
#بالنسبة لعمود ال Age
#ممكن يكون افضل تعويض لقيم السن هوة متوسط الاعمار كلها على السفينة
# .fillna() :  to replace Null values in dataframe
# .mean() : عشان نجيب متوسط الاعمار
x_test['Age']=x_test['Age'].fillna(value=x_test['Age'].mean())
x_train['Age']=x_train['Age'].fillna(value=x_train['Age'].mean())
x_test.head()
x_train.head()
#بالنسبة لطبقات الشعب والتقسيمة داخل السفينة : ال Emabarked
#عايزين نعوض القيم المفقودة : يوجد قيمتين مفقودتين فى ال train
#وطبعا ماينفعش اجيب المتوسط بتاعهم عشان دول ارقام كباين 
#فافضل حل ليهم هوة انى اجيب اكثر تكرار ليهم
#يعنى مثلا لو ال Q مكررة كتير فهيحط ال Q
# .mode() : is detected by collecting and organizing data to count the frequency of each result.
x_train['Embarked']=x_train['Embarked'].fillna(value=x_train['Embarked'].mode()[0])
#دلوقتى انا عايز اعرف كل المعلومات عن ال train 
x_train.info()
#دلوقتى عايزين نرسم histogram لاى عمود عشان نتاد من الشغل
#.hist(): لعمل رسم بيانى
#bins=int : Bins are the number of intervals you want to divide all of your data into, such that it can be displayed as bars on a histogram. A simple method to work our how many bins are suitable is to take the square root of the total number of values in your distribution
x_train['Survived'].hist(bins=5)
x_train['Pclass'].hist(bins=5)
x_train['Sex'].hist(bins=5)
#دلوقتى لما رسمنا بيانيا ال sex لاقيناة عمل لكل حرف من ال male&female عمود داخل الرسم البيانى بمعنى انة رسم لكل حرف scale عمود على الرسم البيانى ودة طبعا مش عايزينة وهنحلة قدام
#فهنخلى ال male ياخد 0 وال female ياخد 1..بمعنى هنحولهم لارقام عشان يرسمهم بيانيا مظبوط لكل جنس يكون لية عمود داخل الرسم البيانى
x_train['Sex']=pd.factorize(x_train['Sex'])[0]
x_test['Sex']=pd.factorize(x_test['Sex'])[0]
#ونفس الوضع لطبقات المجتمع بالسفينة فى ال Embarked
x_train['Embarked']=pd.factorize(x_train['Embarked'])[0]
x_test['Embarked']=pd.factorize(x_test['Embarked'])[0]
#دلوقتى لو جينا نرسم ال sex تانى عشان يظهروا بشكل واضح وكويس 
x_train['Sex'].hist(bins=20)
#رسم الاحياء اللى فضلوا عايشين على السفينة
x_train['Survived'].hist(bins=20)
#رسم رتب الكبائن
x_train['Pclass'].hist(bins=20)
#رسم الاعمار
x_train['Age'].hist(bins=20)
#رسم عدد مرافقين النساء من الرجال او العكس
x_train['SibSp'].hist(bins=20)
#رسم عدد الاهل بالنسبة للفرد بمعنى عدد الافراد معاهم اهلهم لكل فرد بالسفينة يعنى مثلا ظاهر فى 0 600 فرد بمعنى ان فية حوالى 600 شخص ماكنش معاهم اهلهم
x_train['Parch'].hist(bins=20)
#رسم كام طبقة مجتمع بالسفينة وبيان انة اكبر طبقة كانت موجودة
x_train['Embarked'].hist(bins=20)
#هرسم figure عشان احط فية مقارناتى (ارقام ال x وال y) فى الرسم البيانى 
fig=plt.figure(figsize=(10,10))
#هنا انا محتاج ارسم علاقة بين ال survived وال pclass وبقولة روح لل x_train فى ال location اللى فية survived اللى فية رقم 1 وقارنة بال pclass
sns.distplot(x_train.loc[x_train['Survived']==1]['Pclass'])
#هنا انا محتاج ارسم علاقة بين ال survived وال pclass وبقولة روح لل x_train فى ال location اللى فية survived اللى فية رقم 0 وقارنة بال pclass
sns.distplot(x_train.loc[x_train['Survived']==0]['Pclass'])
#دلوقتى احنا أظهرنا نسبة النجاة فى طبقات المجتمع بالسفينة
#وظهر لينا دلوقتى بعد تشغيل الكود ان نسبة النجاة فى الطبقات العليا - 3 - كانت عالية
#طيب دلوقتى انا مش عارف الاوان اللى عندى ممثلة لمين الاحياء ولا الاموات..وانا محتاج اعرف فهمنعمل label للرسمة البيانية
sns.distplot(x_train.loc[x_train['Survived']==1]['Pclass'],kde_kws={'label':'Survived'})
sns.distplot(x_train.loc[x_train['Survived']==0]['Pclass'],kde_kws={'label':'died'})
#########
#ونفس النظام هنعمل رسمة تبين العلاقة بين ال Age وال Pclass
sns.distplot(x_train.loc[x_train['Survived']==1]['Age'],kde_kws={'label':'Survived'})
sns.distplot(x_train.loc[x_train['Survived']==0]['Age'],kde_kws={'label':'died'})
#كدة فهمنا ان اكثر اللى عاشوا كانوا فى سن مابين 20 و 40 واكثر اللى ماتوا بردك كانوا مابين 20 و 40
#عرفنا بردك ان نسبة الاطفال مابين سن ال 0 وال 20 اكثرهم احياء بمعنى ان فى الغالب الاهل كانوا بيضحوا عشان اطفالهم يعيشوا
#عرفنا بردك ان الل فوق سن ال 60 اغلبهم بنسبة كبيرة جدا ماتوا لان نسبة الاحيا عندهم ضعيفة جدا
#########
#طيب تعالوا نشوف علاقة الاحياء والاموات مع الجنس
sns.distplot(x_train.loc[x_train['Survived']==1]['Sex'],kde_kws={'label':'Survived'})
sns.distplot(x_train.loc[x_train['Survived']==0]['Sex'],kde_kws={'label':'died'})
#دلوقتى عشان نفهم الرسمة محتاجين نفتكر مين ال 0 ومين ال 1
#الرجال كانوا ال 0 والنساء كانوا ال 1
#وطبعا فى ال label فاللون البرتقالى هوة عد الوفيات واللون الازرق هوة عدد الاحياء
#نفهم كدة من الرسمة ان اللى على اليمين دول هما النساء, ونفهم من كدة ان نسبة الوفيات عندهم كانت كبيرة جدا وصلت لاقصى الرسمة البيانية ومقارن بنسبة الوفيات عند الرجال على شمال الرسمة البيانية كانوا اقل بكثير من النساء
#اما بالنسبة للاحياء فنسبة الاحياء بردك عند النساء كانوا اكبر من الرجال بس بنسبة صغيرة
#################
#طيب تعالوا شوف دلوقتى علاقة الاحياء والاموات بالرجال اللى كانوا معاهم نساء والعكس
sns.distplot(x_train.loc[x_train['Survived']==1]['SibSp'],kde_kws={'label':'Survived'})
sns.distplot(x_train.loc[x_train['Survived']==0]['SibSp'],kde_kws={'label':'died'})
#نفهم كدة من الرسمة ان الاشخاص اللى ماكنوش معاهم اى مرافق ( 0 مرافق) كانوا فى المقام الاول من احياء او اموات (ولكن اكثرهم كانوا اموات وتقريبا اكبر نسبة وفاه فيهم وبردك اكبر نسبة نجاة فيهم)
#فى المقام اللى بعده عالطول الاشخا اللى كان معاهم مرافق 1 فقط كانت نسبة النجاة فيهم كبيرة جدا مقارنة بنسبة الوفاه
#بعدهم فى المقام اللى كان معاهم 2 او 3 او 4 او 5 او 6 او 7 او 8 او اكثر مرافق..فنسبة الوفاة اعلى من الاحساء بس من بعد الاسخاص اللى معاهم 4 مرافق اغلب الناس ماتوا مافيش حد فيهم عاش
######################
#طيب تعالوا نشوف علاقة الناس اللى كانوا معاهم اقارب بالاحياء والاموات
sns.distplot(x_train.loc[x_train['Survived']==1]['Parch'],kde_kws={'label':'Survived'})
sns.distplot(x_train.loc[x_train['Survived']==0]['Parch'],kde_kws={'label':'died'})
#دلوقتى ظهر لينا ان اللى ماكنش معاه اى اقارب عدد الوفاة كانت عالية جدا ونسبة الحياة بردك كانت عالية(تقريبا اكبر نسبة حياة واكبر نسبة وفاة)
#افى المقام اللى بعدهم عالطول كانت الناس اللى كانوا معاهم اقارب واحد فقط ..ولكن نسبة الوفاة بالنسبة للاحياء كانت كبرة جدا
#اللى كان معاهم 2 اقارب فى المقام الثالث نسبة الوفاة برجك كانت اعلى من الحياة
#بعدهم فى المقام كانوا اللى معاهم اقارب عدد 3 او 4 او 5 او 6 كلهم فى الغالب ماتوا
################
#تعالوا بقى نشوف عدد الاحياء والاموات بالنسبة للكبائن اللى كانت فى السفينة
sns.distplot(x_train.loc[x_train['Survived']==1]['Embarked'],kde_kws={'label':'Survived'})
sns.distplot(x_train.loc[x_train['Survived']==0]['Embarked'],kde_kws={'label':'died'})
#كدة ظهر لينا ان الناس اللى كانت فى الطبقة الاولى نسبة الوفاة عندهم كانت عاليا
#ولكن فى المقام الاول فى الوفيات كانت الناس اللى فى كبائن الدرجة الثانية بعدهم الدرجة الاولى بعدهم الدرجة الثانية
#فى المقام الاول للاحياء (اللون الازرق) كانت الناس اللى كانوا فى كبائن الدرجة الاولى بعديهم الدرجة الثانية بعدهم الدرجة الثالثة
#هنعمل حاجة (مش شايف ان ليها اى لازمة بس الدكتور عملها فى الفيديو بتاعة) كدة فى موضوع ال Age
#ال Age عندى فى السفينة بيتراوح من 1 ل الى حوالى 80 سنة 
#هنقسمهم الى مجموعات: بمعنى من 0 الى 20 مجموعة واحد ةمن 20 الى 40 مجموعة اثنين وهكذا 
# .bins=4 : معناها انى هقولى قسملى العمود الى 4 مجوعات
x_train['Age']=pd.cut(x_train['Age'],bins=4)
x_test['Age']=pd.cut(x_test['Age'],bins=4)
#طبعا دلوقتى لو حاولت ارسم رسم بيانى لل Age مش هيعرف يرسمهم عشان ال range اللى اتعمل فى ال Age
#فالحل انى اعمل categorize للمجموعات: بمعنى انى هخلى المجموعة الاولى  مثلا (من 0 ل 20) هتاخد 0 والمجموعة الثانية (من 20 ل 30) هتاخد واحد وهكذا
x_train['Age']=pd.factorize(x_train['Age'])[0]
x_test['Age']=pd.factorize(x_test['Age'])[0]
##########################################
#دلوقتى احنا خلصنا جزء تحليل البيانات واظهارها وترتيبها
#محتاجين بقى نشتغل فى ال regressions عشان اشوف البينات بتاعتى وابدا اتنبئ بالنتائج
#بس طبعا محتاج اشيل ال column بتاع ال survived عشان هوة اساس التنبوء
#عشان بعد ما اعمل ال regression بتاعى ابقى عارف مين هيعيش ومين هيموت وابدا اقارن نتياجى بين ال output and input
######################
#دلوقتى هناخد كوبى من ال survived من ال x_train ونحطها فى ال output بتاعى وهنسمية y_train
x_train.info()
x_train.head()
y_train=x_train['Survived']
#دلوقتى محتاجين نشيلها من ال x_train عشان ماتلخبطناش
x_train=x_train.drop('Survived',axis=1)
x_train.info()
#دلوقتى بعد ماظبطنا ال data هنبدا نشتغل فى خوارزميات ال machine-learning
#هنعمل دلوقتى ال logistic regression : وهو بيحاول يطلع علاقة خطية مابين الداتا..بمعنى انة بيفصل ابين الداتا بتاعتى وبيرسم خط
from sklearn.preprocessing import StandardScaler
#ال StandardScaler : هيا  object فلازم اعملها class واحنا هنستعملها عشان نعمل transform لل x وال y train
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
#دلوقتى اتحولت الداتا ل regression نسبا اعرف اتنبئ بيها
#دلوقتى هنستعمل ال logistic regression عشان ارسم الخط بتاعى
from sklearn.linear_model import LogisticRegression
#random stat: عشان اثبت النتايج..بمعنى انى هيفضل يشتغل على عمود واحد
regressor1=LogisticRegression(random_state=0)
#دلوقتى هنمرن الداتا
regressor1.fit(x_train,y_train)
#دلوقتى هوة اتعلم ازاى يتنبئ بالبيانات..ومحتاجين نتنبئ فعليا بيانات لل x_test ونحزنها فى متغير جديد
y_pred1=regressor1.predict(x_test)
#دلوقتى عايزين نشوف توفعاتة الصحيحة..وطبعا هنقارن مع ال y_test 
#لو شوفنا دلوقتى ال y_test هنلاقى عدد ال servived فيهم كتير اوى وانا محتاج اقارنهم بال y_pred1
#فهمنعمل matrix للموضوع دة
from sklearn.metrics import confusion_matrix
#طبعا لازم نعمل drop للعمود اللى اسمة passengerID فى ال y_test عشان مايبقاش فيها الا ال servived عشان اعرف اعمل مقارنة مع ال y_pred1
y_test=y_test.drop('PassengerId',axis=1)
#دلوقتى هندخل ال y_test وال y_pred1 فى ال matrix عشان نعرف نقارنهم ببعض
cm1=confusion_matrix(y_test,y_pred1)
#دلوقتى عشان نفهم هوة عمل اية..لو فتحنا ال cm1 فى ال variable explorer هنشوف الاتى
#السطر الاول هوة الاموات وظهر كالاّتى:  265 تنبوء صحيح و 1 تنبوء خاطيئ
#السطر الثانى هوة الاحياء وظهر كالاّتى: 145 تنيوء صحيح و 7 تنبوء خاطيئ
#وبالتالى فان نسبة الaccuracy : دقة التنبوء
#هتكون نسبتها كالاّتى:
#مجموع التنبؤات الصحيحة / الكلى= (265+145) / (266+152)
print((265+145)/(266+152))
# نسبة الدقة = % 98 
#لو جينا نشوف بخوارزمية تانية وهى: support vector machine
from sklearn.svm import SVC
regressor2=SVC(kernel='rbf',random_state=0)
regressor2.fit(x_train,y_train)
ypred2=regressor2.predict(x_test)
cms2=confusion_matrix(y_test,ypred2)
#نحسب الدقة وصلت لفين
#مجموع التنبؤات الصحيحة / الكلى= (256+101) / (266+152)
print ((256+101)/(266+152))
#طيب دلوقتى انا عايز اتوقع واعيش اللحظة واشوف شغلى وتوقعاتى وصلت لفين D: 
#هجرب وادخل بياناتى كانى كنت داخل السفينة واشوف كدة بالخوارزميتين اللى مرنتهم وعملتهم واشوف كنت هعيش ولا هموت
#لاول لازم نعرف الاصل فى (بعد اخر تعديل طبعا)شيت ال x_train الاعمدة كانت ماشية ازاى عشان مثلا فى العمود الاول كان عبارة عن ال pclass وكنا حولناة ل ارقام بمعنى 0 رتبة كابينة الاولى و 1 رتبة الكابينة الثانية وهكذا
#العمود الثانى كان ال sex وكان 0 لو رجل و 1 لو مراة
#العمود الثالث كان sibsp عدد المرافقين وكنا حولناة لارقام بمعنى 0 =ولا مرافق و 1 =مرافق واحد وهكذا 
#العمود الثالث كان parch عدد الاقارب وكنا حولناه لارقام بمعنى 0= ولا قريب و 1 = قريب واحد وهكذا
#العمود الرابع كان Embarked رتبة الكابينة بمعنى 0 = رتبة كابينة درجة اولى و 1 = رتبة كابينة درجة ثانية وهكذا

#وطبعا ماننساش ان regressor1 مستعملة فى ال logisticregression درجة دقة %98
#وطبعا ماننساش ان regressor2 مستعملة فى ال S.vector machine درجة دقة %85
#دلوقتى هندخل بياناتى (على حسب ترتيب الاعمدة طبعا)
#وهنعمل reshape عشان مايفهمش انى عامل ليست
mynum=np.array([1,1,32,1,2,2]).reshape(1,-1)
ypred3=regressor1.predict(mynum)
ypred4=regressor2.predict(mynum)