import os
import uuid
import sqlite3
import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------- Flask Config --------------------
app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Database Setup --------------------
DB_NAME = "users.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password TEXT NOT NULL
        )
        """)
        conn.commit()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# -------------------- Model Setup --------------------
img_size = (224, 224)
model_path = "best_mobilenetv2_model.h5"  # Path to your trained model
model = load_model(model_path)
print("✅ Model loaded successfully.")

valid_classes = [
    "Aloe_vera", "Arjuna", "Ashwagandha", "Basil", "Bay_leaf",
    "Calendula", "Chamomile", "Cinnamon", "Elderberry", "Eucalyptus",
    "Fennel", "Garlic", "Ginkgo_biloba", "Gotu_Kola", "Hibiscus",
    "Lavender", "Moringa", "Neem", "Rosemary", "Sage",
    "Shankhpushpi", "St_Johns_Wort", "Tulsi", "Turmeric", "Valerian"
]

# ✅ Paste your full medicinal_uses dictionary here
medicinal_uses = {
    "Aloe_vera": """Heals burns and cuts – Aloe vera gel soothes and accelerates healing of minor burns, wounds, and sunburns.

Reduces skin inflammation – Helps treat conditions like eczema, psoriasis, and rashes due to its anti-inflammatory properties.

Acts as a natural moisturizer – Hydrates the skin without clogging pores, making it ideal for all skin types.

Improves digestive health – Aloe vera juice can relieve constipation, acidity, and symptoms of irritable bowel syndrome.

Supports wound healing – Its antibacterial and antifungal effects prevent infection and promote faster tissue repair.

Controls blood sugar – May help lower blood glucose levels and improve insulin sensitivity under medical guidance.

Boosts immunity – Contains antioxidants and enzymes that detoxify the body and strengthen the immune system.

Promotes hair growth – Reduces dandruff, nourishes the scalp, and supports healthier, thicker hair.

Soothes oral problems – Aloe vera mouthwash can reduce gum inflammation and plaque buildup.

Detoxifies the body – Helps flush out toxins and maintain overall wellness when taken in moderation.""",

    "Arjuna": """Supports heart health – Arjuna bark is traditionally used to strengthen the heart muscles and improve cardiac function.

Controls blood pressure – It helps regulate high blood pressure and maintain healthy circulation.

Reduces cholesterol – Arjuna may lower LDL (bad) cholesterol and improve overall lipid profile.

Acts as a natural antioxidant – Protects heart tissues from oxidative stress and damage.

Improves angina symptoms – Often used in Ayurveda to relieve chest pain and improve blood flow to the heart.

Aids in wound healing – Its bark has anti-inflammatory and antibacterial properties that help heal external wounds.

Strengthens bones – Traditionally believed to support bone health due to its mineral content.

Enhances respiratory health – Used in herbal medicine to manage asthma, cough, and other respiratory issues.

Supports liver health – Helps detoxify and protect the liver from damage caused by toxins.

Boosts overall stamina – Improves energy, reduces stress on the cardiovascular system, and promotes vitality.""",

    "Ashwagandha": """Reduces stress and anxiety – Ashwagandha is well known as an adaptogen that lowers cortisol levels and promotes calmness.

Boosts energy and stamina – Improves physical endurance and helps combat fatigue.

Enhances brain function – Supports memory, focus, and cognitive performance while reducing mental fog.

Improves sleep quality – Traditionally used to treat insomnia and promote deeper, restorative sleep.

Supports immunity – Strengthens the immune system and helps the body fight infections.

Balances hormones – Aids in regulating thyroid function and reproductive hormones.

Supports heart health – May reduce blood pressure and cholesterol, improving overall cardiovascular health.

Increases muscle strength – Helps build muscle mass and improve recovery after exercise.

Stabilizes blood sugar – Can help lower blood glucose levels when taken regularly under medical supervision.

Slows aging – Rich in antioxidants that protect cells from damage and support longevity.""",
    
    "Basil":"""Boosts immunity – Basil leaves strengthen the immune system and help the body fight infections.

Relieves respiratory problems – Effective in managing cough, cold, asthma, and bronchitis due to its expectorant properties.

Reduces stress – Acts as a natural adaptogen that lowers stress and promotes mental clarity.

Supports heart health – Helps regulate blood pressure and cholesterol, improving cardiovascular wellness.

Improves digestion – Enhances appetite, reduces bloating, and aids in smooth digestion.

Fights inflammation – Contains eugenol and other compounds with strong anti-inflammatory effects.

Controls blood sugar – May help maintain stable blood glucose levels under medical supervision.

Protects against infections – Its antibacterial and antifungal properties combat harmful microbes.

Promotes oral health – Chewing basil leaves or using its extract helps prevent gum disease and bad breath.

Rich in antioxidants – Neutralizes harmful free radicals, protecting cells and slowing aging. """,
    
    "Bay_leaf":""" Supports digestion – Bay leaves help relieve bloating, indigestion, and flatulence.

Manages diabetes – Can help regulate blood sugar levels and improve insulin function under medical supervision.

Boosts immunity – Rich in vitamin C and antioxidants that strengthen the immune system.

Improves heart health – Contains compounds that may reduce cholesterol and support healthy blood pressure.

Relieves respiratory problems – Steam inhalation with bay leaves can ease cough, cold, and congestion.

Fights inflammation – Eugenol and other phytonutrients help reduce inflammation and pain.

Aids in wound healing – Has antimicrobial and antifungal properties to protect against infections.

Promotes relaxation – The aroma of bay leaves is traditionally used to reduce stress and improve sleep quality.

Supports kidney health – Acts as a mild diuretic, helping the body eliminate excess fluids and toxins.

Improves skin health – Bay leaf oil or paste can help reduce acne, infections, and skin irritation.""",
    
    "Calendula":""" Heals wounds and cuts – Calendula ointments and creams speed up healing of minor cuts, scrapes, and burns.

Soothes skin irritation – Reduces redness, itching, and inflammation from eczema, rashes, or insect bites.

Fights infections – Its natural antibacterial and antifungal properties help prevent wound infections.

Improves oral health – Calendula mouth rinses can ease gum inflammation, mouth ulcers, and sore throats.

Relieves diaper rash – Calendula creams are widely used to soothe and protect babies’ sensitive skin.

Supports digestive health – Tea made from calendula flowers may relieve gastritis, ulcers, and digestive inflammation.

Boosts immunity – Rich in antioxidants and flavonoids that strengthen the immune system.

Reduces menstrual discomfort – Traditionally used to ease cramps, regulate periods, and balance hormones.

Calms sunburns – Helps cool and repair skin damaged by sun exposure.

Promotes healthy skin – Enhances skin tone, reduces acne, and helps fade scars naturally. """,
    
    "Chamomile":"""Promotes relaxation – Chamomile tea is famous for reducing stress, anxiety, and promoting calmness.

Improves sleep quality – Acts as a gentle sedative, helping people fall asleep faster and sleep more deeply.

Soothes digestive problems – Relieves indigestion, bloating, gas, and mild stomach cramps.

Reduces inflammation – Its anti-inflammatory properties help calm irritated tissues internally and externally.

Heals skin conditions – Chamomile compresses or creams soothe eczema, rashes, sunburn, and minor wounds.

Supports oral health – Chamomile mouth rinse can ease gum inflammation, mouth sores, and sore throats.

Relieves menstrual discomfort – Drinking chamomile tea may reduce menstrual cramps and mood swings.

Boosts immunity – Contains antioxidants that strengthen the body’s defense against infections.

Eases cold symptoms – Inhaling chamomile steam can help clear nasal congestion and soothe respiratory irritation.

Protects hair and scalp – Chamomile rinse can reduce dandruff, calm an itchy scalp, and add shine to hair. """,
    
    "Cinnamon":"""Regulates blood sugar – Cinnamon helps improve insulin sensitivity and manage blood glucose levels.

Supports heart health – May reduce LDL (bad) cholesterol and triglycerides, improving cardiovascular wellness.

Fights infections – Its antimicrobial and antifungal properties help combat harmful bacteria and fungi.

Reduces inflammation – Contains cinnamaldehyde and antioxidants that lower inflammation in the body.

Improves digestion – Stimulates digestive enzymes, relieves bloating, and eases indigestion.

Boosts immunity – Rich in antioxidants that strengthen the immune system against infections.

Relieves cold and cough – Cinnamon tea can soothe sore throats, clear congestion, and ease respiratory discomfort.

Enhances brain function – Compounds in cinnamon may support memory and concentration.

Promotes oral health – Its antibacterial properties help prevent gum disease and bad breath.

Acts as a natural preservative – Traditionally used to preserve foods and protect against spoilage due to its antimicrobial action. """,
    
    "Elderberry":"""Boosts immunity – Elderberries are rich in antioxidants and vitamin C that strengthen the immune system.

Fights colds and flu – Elderberry syrup is traditionally used to shorten the duration and severity of colds and influenza.

Reduces inflammation – Anthocyanins in elderberries have strong anti-inflammatory effects.

Supports respiratory health – Helps relieve sinus congestion, sore throat, and cough.

Improves heart health – May help lower cholesterol, reduce blood pressure, and improve circulation.

Promotes healthy skin – Antioxidants protect skin from free radical damage and support a youthful appearance.

Supports digestive health – Elderberries have mild laxative effects that can relieve constipation.

Helps fight infections – Contains antiviral and antibacterial compounds that inhibit the growth of harmful pathogens.

Aids in weight management – Low in calories but high in nutrients, supporting healthy metabolism.

Protects against oxidative stress – High levels of flavonoids neutralize free radicals and reduce cell damage. """,

    "Eucalyptus":"""Clears respiratory congestion – Eucalyptus oil steam inhalation helps relieve colds, sinusitis, and nasal congestion.

Soothes cough and sore throat – Commonly used in lozenges, teas, and vapors to ease throat irritation.

Acts as a natural decongestant – Its active compound, cineole (eucalyptol), helps open airways and improve breathing.

Relieves muscle and joint pain – Eucalyptus oil massage can reduce pain from arthritis, sprains, and soreness.

Fights infections – Has antibacterial and antifungal properties that protect against harmful microbes.

Promotes oral health – Eucalyptus extract in mouthwashes reduces plaque, gingivitis, and bad breath.

Reduces inflammation – Its anti-inflammatory effects soothe irritated skin, gums, and mucous membranes.

Repels insects – Eucalyptus oil is a natural insect repellent and helps treat insect bites.

Boosts mental clarity – The refreshing aroma may improve focus, alertness, and reduce mental fatigue.

Supports wound healing – Diluted eucalyptus oil can be applied to minor cuts to prevent infection and speed healing. """,

    "Fennel":"""Improves digestion – Fennel seeds help relieve bloating, gas, indigestion, and constipation.

Relieves menstrual discomfort – Can reduce cramps and regulate menstrual cycles.

Boosts lactation – Traditionally used to increase milk production in breastfeeding mothers.

Supports respiratory health – Helps clear mucus, ease cough, and soothe sore throats.

Reduces inflammation – Contains antioxidants that lower inflammation in the body.

Controls blood pressure – Fennel’s potassium content helps regulate high blood pressure.

Promotes oral health – Chewing fennel seeds freshens breath and prevents oral infections.

Supports weight management – Acts as a mild diuretic and helps control appetite.

Enhances metabolism – Helps in better absorption of nutrients and promotes overall metabolic health.

Protects against oxidative stress – Rich in flavonoids and phenolic compounds that neutralize free radicals. """,

    "Garlic":""" Boosts immunity – Garlic strengthens the immune system and helps fight infections.

Reduces blood pressure – Allicin in garlic can help lower high blood pressure naturally.

Improves heart health – Helps reduce cholesterol and triglycerides, supporting cardiovascular wellness.

Acts as a natural antibiotic – Has antibacterial, antiviral, and antifungal properties.

Supports respiratory health – Can relieve coughs, colds, and respiratory infections.

Controls blood sugar – Helps regulate glucose levels, beneficial for people with diabetes.

Reduces inflammation – Contains antioxidants that reduce inflammation and oxidative stress.

Promotes detoxification – Aids liver function and helps remove toxins from the body.

Supports bone health – May improve bone density and reduce the risk of osteoporosis.

Enhances athletic performance – Can reduce fatigue and improve endurance in physical activities.""",

    "Ginkgo_biloba":""" Improves memory and cognitive function – Enhances concentration, learning, and memory retention.

Supports brain health – Increases blood flow to the brain, helping prevent age-related cognitive decline.

Reduces anxiety and stress – Acts as a natural adaptogen to calm nerves and improve mood.

Supports eye health – Improves circulation to the eyes and may help prevent macular degeneration.

Improves circulation – Strengthens blood vessels and promotes healthy blood flow throughout the body.

Reduces symptoms of dementia – May help alleviate mild memory loss and confusion in early dementia.

Acts as an antioxidant – Protects cells from oxidative stress and free radical damage.

Supports heart health – Improves blood vessel function and may reduce the risk of cardiovascular issues.

Relieves tinnitus – Can help reduce ringing in the ears caused by poor circulation.

Supports respiratory health – Helps reduce inflammation and improve oxygen utilization in lungs.""",

    "Gotu_Kola":""" Improves wound healing – Stimulates collagen production and accelerates the repair of cuts, burns, and scars.

Boosts cognitive function – Enhances memory, focus, and overall brain function.

Reduces anxiety and stress – Acts as a natural adaptogen to calm the nervous system.

Supports circulation – Strengthens veins and capillaries, helping prevent varicose veins and swelling.

Promotes skin health – Reduces stretch marks, cellulite, and signs of aging due to its antioxidant properties.

Enhances liver function – Supports detoxification and overall liver health.

Aids in digestive health – Can relieve stomach ulcers and improve intestinal function.

Reduces inflammation – Helps manage arthritis and other inflammatory conditions.

Supports respiratory health – Traditionally used to relieve asthma, cough, and bronchitis.

Boosts immunity – Contains compounds that strengthen the body’s natural defense mechanisms.""",

    "Hibiscus":""" Lowers blood pressure – Hibiscus tea may help reduce high blood pressure naturally.

Supports heart health – Helps lower cholesterol and triglyceride levels, promoting cardiovascular wellness.

Boosts immunity – Rich in vitamin C and antioxidants that strengthen the immune system.

Aids digestion – Can relieve constipation, improve bowel movement, and support overall digestive health.

Promotes liver health – Helps detoxify the liver and improve its function.

Reduces inflammation – Contains compounds that help reduce swelling and inflammation in the body.

Supports weight management – May help in reducing body fat and controlling obesity.

Improves skin health – Antioxidants in hibiscus protect against aging, acne, and skin damage.

Relieves menstrual discomfort – Can ease cramps and regulate menstrual cycles.

Acts as a natural diuretic – Helps remove excess fluids, reducing bloating and water retention.""",

    "Lavender":"""Reduces stress and anxiety – Lavender aromatherapy calms the mind and promotes relaxation.

Improves sleep quality – Helps treat insomnia and encourages deeper, restorative sleep.

Relieves headaches – Inhalation or topical application can reduce tension and migraine headaches.

Supports skin health – Has antiseptic and anti-inflammatory properties that help heal burns, cuts, and acne.

Eases muscle pain – Lavender oil massage can relieve soreness, cramps, and joint pain.

Boosts respiratory health – Inhalation helps relieve cough, cold, and sinus congestion.

Improves mood – Lavender aromatherapy can reduce symptoms of depression and uplift mood.

Promotes wound healing – Its antibacterial properties prevent infections and accelerate healing of minor wounds.

Supports hair health – Can reduce dandruff, improve scalp health, and promote hair growth.

Acts as a natural insect repellent – Helps repel mosquitoes and other insects naturally. """,

    "Moringa":"""Rich in nutrients – Provides vitamins, minerals, and amino acids essential for overall health.

Boosts immunity – Contains antioxidants and compounds that strengthen the immune system.

Reduces inflammation – Helps manage conditions like arthritis and other inflammatory disorders.

Supports heart health – May reduce cholesterol and blood pressure, promoting cardiovascular wellness.

Regulates blood sugar – Can help lower blood glucose levels in people with diabetes.

Improves digestion – Aids in relieving constipation, bloating, and other digestive issues.

Promotes healthy skin – Antioxidants in Moringa reduce signs of aging, acne, and skin damage.

Supports liver health – Protects the liver from toxins and improves its function.

Enhances energy and stamina – Boosts physical performance and reduces fatigue.

Supports brain health – Contains nutrients that improve memory, focus, and cognitive function.""",

    "Neem":"""Supports skin health – Neem treats acne, eczema, psoriasis, and other skin infections.

Acts as an antimicrobial – Has antibacterial, antiviral, and antifungal properties.

Boosts immunity – Strengthens the body’s natural defense against infections.

Supports oral health – Neem sticks or extracts help prevent gum disease, cavities, and bad breath.

Regulates blood sugar – Can help control blood glucose levels in diabetes.

Reduces inflammation – Effective in alleviating joint pain, arthritis, and other inflammatory conditions.

Detoxifies the body – Helps cleanse blood and remove toxins naturally.

Supports liver health – Protects the liver from damage and improves its function.

Repels insects – Neem oil acts as a natural insect repellent and treats insect bites.

Promotes hair health – Reduces dandruff, strengthens hair roots, and prevents scalp infections. """,

    "Rosemary":""" Improves memory and concentration – Rosemary aroma and extracts support brain function and alertness.

Boosts immunity – Rich in antioxidants that strengthen the body’s defense system.

Supports digestive health – Helps relieve indigestion, bloating, and gas.

Reduces inflammation – Anti-inflammatory compounds help with arthritis and muscle pain.

Enhances hair health – Stimulates hair growth, reduces dandruff, and improves scalp circulation.

Supports heart health – May improve blood circulation and reduce cholesterol levels.

Relieves respiratory issues – Eases congestion, cough, and asthma symptoms.

Acts as an antimicrobial – Protects against bacteria and fungi, supporting overall health.

Promotes skin health – Helps in healing wounds, acne, and improving skin tone.

Boosts mood and reduces stress – Aromatherapy with rosemary can relieve anxiety and improve mental well-being.""",

    "Sage":""" Improves brain function – Enhances memory, focus, and cognitive performance.

Reduces inflammation – Contains anti-inflammatory compounds that help with arthritis and sore muscles.

Supports digestive health – Relieves bloating, indigestion, and stomach cramps.

Fights infections – Has antibacterial, antiviral, and antifungal properties.

Relieves sore throat – Sage tea or gargle can soothe throat irritation and cough.

Balances blood sugar – May help regulate blood glucose levels in diabetes.

Supports oral health – Helps reduce gum disease, bad breath, and oral inflammation.

Boosts immunity – Rich in antioxidants that strengthen the body’s defense system.

Promotes skin health – Can help with acne, eczema, and minor wounds due to its healing properties.

Reduces hot flashes and menopause symptoms – Traditionally used to ease hormonal imbalances in women.""",

    "Shankhpushpi":""" Enhances memory and learning – Traditionally used to improve cognitive function and retention.

Reduces stress and anxiety – Acts as a natural adaptogen, calming the nervous system.

Improves sleep quality – Helps treat insomnia and promotes deep, restorative sleep.

Supports brain health – Protects neurons from damage and slows age-related cognitive decline.

Boosts concentration and focus – Useful for students and people with mental fatigue.

Acts as a mild sedative – Helps relax the mind and reduce hyperactivity.

Supports mental clarity – Reduces mental fog and enhances alertness.

Helps manage depression – Traditionally used in Ayurveda to improve mood and emotional balance.

Protects against oxidative stress – Rich in antioxidants that neutralize free radicals in the brain.

Supports overall nervous system health – Strengthens nerves and improves overall neurological function.""",

    "St_Johns_Wort":""" Reduces depression – Widely used as a natural remedy for mild to moderate depression.

Alleviates anxiety – Helps calm nervous tension and reduce stress levels.

Supports sleep – Can improve sleep quality and reduce insomnia.

Heals wounds – Topical application promotes healing of cuts, burns, and bruises.

Reduces inflammation – Has anti-inflammatory properties that soothe skin and tissue irritation.

Relieves nerve pain – May help manage neuropathic pain, sciatica, and nerve-related discomfort.

Supports mood balance – Helps regulate mood swings and emotional instability.

Protects against oxidative stress – Contains antioxidants that neutralize free radicals.

Improves skin health – Can be used to treat minor skin irritations, rashes, and sunburn.

Aids in seasonal affective disorder (SAD) – Traditionally used to reduce symptoms of seasonal depression.""",

    "Tulsi":""" Boosts immunity – Tulsi strengthens the immune system and helps fight infections.

Reduces stress and anxiety – Acts as a natural adaptogen, calming the mind and reducing cortisol levels.

Supports respiratory health – Helps relieve cough, cold, asthma, and congestion.

Controls blood sugar – May help regulate glucose levels in people with diabetes.

Improves heart health – Supports healthy blood pressure and cholesterol levels.

Promotes digestive health – Relieves indigestion, bloating, and acidity.

Fights inflammation – Contains compounds that reduce inflammation in the body.

Protects liver health – Helps detoxify the liver and improve its function.

Enhances skin health – Can prevent acne, skin infections, and premature aging.

Supports mental clarity – Improves focus, memory, and overall cognitive function.""",

    "Turmeric":"""Reduces inflammation – Curcumin, the active compound, helps manage arthritis, joint pain, and inflammatory conditions.

Acts as an antioxidant – Neutralizes free radicals and protects cells from oxidative stress.

Supports heart health – Helps lower cholesterol and improves blood vessel function.

Boosts immunity – Strengthens the body’s defenses against infections and diseases.

Promotes liver health – Aids detoxification and protects the liver from damage.

Improves digestion – Helps relieve bloating, gas, and digestive discomfort.

Supports skin health – Reduces acne, scars, and other skin inflammations.

May prevent cancer – Curcumin has shown potential in inhibiting the growth of certain cancer cells.

Reduces symptoms of depression – May improve mood and alleviate mild depression.

Supports brain health – Enhances memory, focus, and may protect against neurodegenerative disorders. """,

    "Valerian" : """Improves sleep quality – Valerian root is widely used to treat insomnia and promote deep, restful sleep.

Reduces anxiety and stress – Acts as a natural sedative to calm the nervous system.

Eases restlessness – Helps relax the body and mind, reducing irritability and tension.

Supports mental clarity – Can improve focus and reduce mental fatigue.

Relieves headaches – May help alleviate tension headaches and migraines.

Reduces muscle cramps and spasms – Acts as a natural relaxant for smooth and skeletal muscles.

Supports heart health – Can help lower blood pressure and promote cardiovascular relaxation.

Assists in menopause symptoms – Helps reduce hot flashes and mood swings.

Acts as a mild pain reliever – May help with minor aches and joint pain.

Supports overall nervous system health – Strengthens the nervous system and promotes calmness. """
    

}

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name, email, phone, password = (
            request.form["name"],
            request.form["email"],
            request.form["phone"],
            request.form["password"],
        )
        hashed_password = generate_password_hash(password)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                (name, email, phone, hashed_password),
            )
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, password = request.form["email"], request.form["password"]
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"], session["user_name"] = user["id"], user["name"]
            flash("Login successful!", "success")
            return redirect(url_for("predict_plant"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/about")
def about():
    return render_template("about.html")

# -------------------- Plant Prediction --------------------
@app.route("/predict", methods=["GET", "POST"])
def predict_plant():
    if "user_id" not in session:
        flash("Please login to access plant prediction.", "warning")
        return redirect(url_for("login"))

    prediction = None
    confidence = None
    uses_text = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # preprocess image
            img = image.load_img(filepath, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # prediction
            preds = model.predict(img_array)
            predicted_index = np.argmax(preds, axis=1)[0]
            prediction = valid_classes[predicted_index]
            confidence = round(preds[0][predicted_index] * 100, 2)
            uses_text = medicinal_uses.get(prediction, "No medicinal uses available.")

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        uses_text=uses_text,
        filename=filename,
    )

@app.route("/get_use/<plant_name>")
def get_use(plant_name):
    use = medicinal_uses.get(plant_name, "No data available for this plant.")
    return jsonify({"plant": plant_name, "uses": use})

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
