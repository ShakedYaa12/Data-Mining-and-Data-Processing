import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from car_data_prep import prepare_data


app = Flask(__name__)
model = pickle.load(open("./models/trained_model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html', form_data={}, prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    required_columns = ['manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine',
                        'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Pic_num', 'Cre_date', 'Repub_date',
                        'Description', 'Color', 'Km', 'Test', 'Supply_score']

    data = {col: request.form.get(col, None) for col in required_columns}

    features_df = pd.DataFrame([data])

    # המרת ערכים לטיפוסים נכונים
    features_df['Year'] = pd.to_numeric(features_df['Year'], errors='coerce')
    features_df['Hand'] = pd.to_numeric(features_df['Hand'], errors='coerce')

    # עיבוד הנתונים לפני שליחה למודל
    prepared_df = prepare_data(features_df)

    # רשימת כל העמודות המצופה לאחר האנקודד
    expected_columns = ['manufactor_אאודי', 'manufactor_אופל', 'manufactor_אלפא רומיאו', 'manufactor_ב.מ.וו',
                        'manufactor_דייהטסו', 'manufactor_הונדה', 'manufactor_וולוו', 'manufactor_טויוטה',
                        'manufactor_יונדאי', 'manufactor_לקסוס', 'manufactor_מאזדה', 'manufactor_מיני',
                        'manufactor_מיצובישי', 'manufactor_מרצדס', 'manufactor_ניסאן', 'manufactor_סובארו',
                        'manufactor_סוזוקי', 'manufactor_סיטרואן', 'manufactor_סקודה', 'manufactor_פולקסווגן',
                        'manufactor_פורד', "manufactor_פיג'ו", 'manufactor_קיה', 'manufactor_קרייזלר',
                        'manufactor_רנו', 'manufactor_שברולט', 'model_106', 'model_108', 'model_120i',
                        'model_159', 'model_2', 'model_200', 'model_2008', 'model_208', 'model_220', 'model_25',
                        'model_3', 'model_300C', 'model_301', 'model_307CC', 'model_308', 'model_316', 'model_318',
                        'model_320', 'model_325', 'model_5', 'model_5008', 'model_508', 'model_523', 'model_525',
                        'model_530', 'model_6', 'model_A1', 'model_A3', 'model_A4', 'model_A5', 'model_A6',
                        'model_ASX', 'model_AX', 'model_All Road', 'model_B3', 'model_B4', 'model_C-CLASS קופה',
                        'model_C-Class', 'model_C-Class Taxi', 'model_C-Class קופה', 'model_C-HR', 'model_C1',
                        'model_C3', 'model_C30', 'model_C4', 'model_C5', 'model_CADDY COMBI', 'model_CLK',
                        'model_CT200H', 'model_CX', 'model_DS3', 'model_E- CLASS', 'model_E-Class',
                        'model_E-Class קופה / קבריולט', 'model_FR-V', 'model_GS300', 'model_GT3000',
                        'model_INSIGHT', 'model_IS250', 'model_IS300H', 'model_IS300h', 'model_JAZZ',
                        'model_M1', 'model_ONE', 'model_Q3', 'model_R8', 'model_RC', 'model_RCZ',
                        'model_RS5', 'model_S-Class', 'model_S3', 'model_S5', 'model_S60', 'model_S7',
                        'model_S80', 'model_SLK', 'model_SVX', 'model_SX4', 'model_SX4 קרוסאובר',
                        'model_V- CLASS', 'model_V40', 'model_V40 CC', 'model_X1', 'model_XCEED', 'model_XV',
                        'model_i10', 'model_i20', 'model_i25', 'model_i30', 'model_i30CW', 'model_i35', 'model_one',
                        'model_אאוטבק', 'model_אאוטלנדר', 'model_אדם', 'model_אודסיי', 'model_אוונסיס',
                        'model_אונסיס', 'model_אוקטביה', 'model_אוקטביה קומבי', 'model_אוריס', "model_אטראז'",
                        'model_איגניס', 'model_איוניק', 'model_אימפרזה', 'model_אינסיגניה', 'model_אינסייט',
                        'model_אלטו', 'model_אלמרה', 'model_אלנטרה', 'model_אלתימה', 'model_אס-מקס', 'model_אסטרה',
                        'model_אפלנדר', 'model_אקורד', 'model_אקליפס', 'model_בלנו', "model_ג'אז הייבריד",
                        "model_ג'ולייטה", "model_ג'וק", "model_ג'טה", 'model_ג`אז', 'model_גולף', 'model_גולף GTI',
                        'model_גולף פלוס', 'model_גלאקסי', "model_גראנד, וויאג'ר", 'model_גראנד, וויאג`ר',
                        'model_גרנד סניק', 'model_גרנדיס', 'model_האצ`בק', 'model_וויאג`ר', 'model_ולוסטר',
                        'model_ורסו', 'model_זאפירה', 'model_חיפושית', 'model_חיפושית חדשה', 'model_טוראן',
                        'model_טראקס', 'model_טריוס', 'model_יאריס', 'model_ייטי', 'model_לאונה', 'model_לג`נד',
                        'model_לנסר', 'model_לנסר הדור החדש', 'model_לנסר ספורטבק', 'model_מאליבו',
                        'model_מגאן אסטייט / גראנד טור', 'model_מוסטנג', 'model_מוקה', 'model_מוקה X',
                        'model_מיטו', 'model_מיקרה', 'model_מקסימה', 'model_מריבה', 'model_נוט', 'model_נירו',
                        'model_נירו EV', 'model_נירו PHEV', 'model_סדן', 'model_סדרה 1', 'model_סדרה 3',
                        'model_סדרה 5', 'model_סוויפט', 'model_סוויפט החדשה', 'model_סול', 'model_סונטה',
                        'model_סופרב', 'model_סטוניק', 'model_סיד', 'model_סיוויק', "model_סיוויק האצ'בק",
                        "model_סיוויק האצ'בק החדשה", 'model_סיוויק הייבריד', 'model_סיוויק סדאן',
                        'model_סיוויק סדאן החדשה', 'model_סיוויק סטיישן', 'model_סיטיגו / Citygo',
                        'model_סיריון', 'model_סלריו', 'model_סנטרה', 'model_ספיה', 'model_ספייס',
                        'model_ספייס סטאר', 'model_ספלאש', 'model_סראטו', 'model_פאביה', 'model_פאביה ספייס',
                        'model_פאסאט', 'model_פאסאט CC', 'model_פולו', 'model_פוקוס', 'model_פורטה',
                        'model_פיאסטה', 'model_פיקנטו', 'model_פלואנס', 'model_פלואנס חשמלי', 'model_פריוס',
                                                'model_פרייד', 'model_פרימרה', 'model_קאונטרימן', 'model_קאמרי', 'model_קאנטרימן',
                        'model_קווסט', 'model_קונקט', 'model_קופה', 'model_קופר', 'model_קורבט', 'model_קורולה',
                        'model_קורסה', 'model_קורסה החדשה', 'model_קורסיקה', 'model_קליאו', 'model_קליאו אסטייט',
                        'model_קליאו דור 4', 'model_קפצ`ור', 'model_קרוז', 'model_קרוסאובר', 'model_קרניבל',
                        'model_ראפיד', 'model_ראפיד ספייסבק', 'model_ריו', 'model_שירוקו', 'model_שרמנט',
                        'Gear_אוטומטית', 'Gear_טיפטרוניק', 'Gear_ידנית', 'Gear_רובוטית', 'Engine_type_בנזין',
                        'Engine_type_גז', 'Engine_type_דיזל', 'Engine_type_היברידי', 'Engine_type_חשמלי',
                        'Engine_type_טורבו דיזל', 'Prev_ownership_יצרן', 'Prev_ownership_לא פרטית',
                        'Prev_ownership_פרטית', 'Year', 'Hand', 'capacity_Engine', 'Km',
                        'Authorized_service', 'KM_Per_Year', 'Engine_Efficiency', 'Vintage_Car']

    # יצירת מילון עם נתונים חדשים
    new_columns = {col: 0 for col in expected_columns}

    # עדכון נתוני ה DataFrame הקיים עם העמודות החסרות
    for col in new_columns:
        if col not in prepared_df.columns:
            prepared_df[col] = new_columns[col]

    # סידור העמודות לפי הסדר המצופה
    prepared_df = prepared_df[expected_columns]

    prediction = int(model.predict(prepared_df)[0] // 100) * 100

    # setting minimum value
    if prediction < 8000:
        prediction = 8000
        
    prediction = "{:,}".format(prediction)

    return jsonify({'prediction': f"{prediction} ש''ח"})
    

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

