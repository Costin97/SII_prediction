1. Încărcarea și prelucrarea seturilor de date

  Încărcarea datelor: Am început prin a încărca fișierele CSV care conține textele și etichetele corespunzătoare utilizând biblioteca pandas, utilă pentru manipularea datelor tabelare. Astfel, am citit seturile de date pentru antrenament și test din fișierele respective.
Curățarea textului: După ce am încărcat textele, am aplicat un proces de curățare pentru a face datele mai uniforme și mai ușor de gestionat pentru model. 
Acesta include:
•	Eliminarea spațiilor suplimentare sau a liniilor goale.
•	Transformarea tuturor literelor în minuscule pentru a evita diferențele dintre cuvinte care sunt, de fapt, aceleași (de exemplu, „Câine” și „câine”).
•	Utilizarea unui lematizator din biblioteca NLTK, care reduce cuvintele la forma lor de bază (de exemplu, „cărți” devine „carte”). Aceasta ajută la reducerea variabilității semantice și la îmbunătățirea performanței modelului, întrucât astfel, modelul recunoaște cuvintele indiferent de forma lor grammaticală.

2. Împărțirea datelor

  Împărțirea seturilor de date: Datele au fost împărțite într-un set de antrenament (80% din total) și un set de validare (20% din total). Setul de validare este important pentru a evalua performanța modelului pe date pe care acesta nu le-a întâlnit înainte, astfel încât să putem verifica dacă modelul se generalizează corect pe date noi.
Codificarea etichetelor: Etichetele textuale, cum ar fi „fake”, „biased” și „true”, au fost transformate în valori numerice (0, 1, 2). Această transformare este necesară, deoarece modelele de învățare automată procesează mai ușor valorile numerice decât textul brut.

3. Crearea seturilor de date pentru Hugging Face

  După prelucrarea datelor, am folosit biblioteca Hugging Face pentru a crea seturi de date compatibile cu modelele lor. Am utilizat funcția Dataset.from_pandas() pentru a transforma datele dintr-un DataFrame Pandas într-un format compatibil cu modelele pre-antrenate din Hugging Face. Astfel, am obținut două seturi de date: unul pentru antrenament și unul pentru validare.

4. Încărcarea și pregătirea modelului

  Tokenizarea textului: Tokenizarea este un proces prin care textul brut este transformat într-o secvență de numere pe care modelul o poate înțelege. Am folosit un tokenizator specific pentru modelul Camembert, care este un model pre-antrenat pe limba franceză, similar cu BERT. Tokenizatorul transformă fiecare cuvânt sau subcuvânt din text într-un indice din vocabularul modelului.
  
  Încărcarea modelului: Am utilizat un model pre-antrenat numit CamembertForSequenceClassification, care este deja antrenat pe o mare cantitate de texte. Am specificat că acest model va avea 3 clase pentru clasificarea textelor: „fake” (0), „biased” (1) și „true” (2). Modelul a fost încărcat din biblioteca Hugging Face și apoi mutat pe GPU pentru a accelera procesul de antrenament.

5. Setarea argumentelor de antrenament

Configurarea antrenamentului: Am setat parametrii de antrenament folosind clasa TrainingArguments. Printre acești parametri se numără:

•	Numărul de epoci: Modelul va parcurge datele de antrenament de 5 ori (5 epoci).
•	Dimensiunea lotului: Am setat dimensiunea lotului (batch size) la 16, ceea ce înseamnă că modelul va procesa câte 16 exemple deodată în fiecare pas de antrenament.
•	Rata de învățare: Rata de învățare a fost setată la 3e-5, o valoare mică care ajută modelul să învețe treptat fără a face pași prea mari care ar putea deteriora performanța.
•	Evaluare și salvare a modelului: Modelul este evaluat la sfârșitul fiecărei epoci pe setul de validare și se salvează automat cel mai bun model (în funcție de performanța sa).

6. Antrenarea modelului

  Antrenarea efectivă: Antrenamentul propriu-zis al modelului a fost realizat folosind clasa Trainer din biblioteca Hugging Face. 
  Evaluarea: După fiecare epocă, am evaluat performanța modelului folosind setul de validare. Evaluarea a inclus măsurători precum acuratețea, precizia, recall-ul și scorul F1.


7. Predicția pe setul de test

  După antrenarea și evaluarea modelului, am folosit setul de test pentru a face predicții pe texte pe care modelul nu le-a văzut anterior. 
Obținerea predicțiilor: După ce am aplicat modelul pe datele de test, am obținut predicțiile pentru fiecare text din setul de test. Pentru a obține eticheta finală, am utilizat funcția torch.argmax(), care a ales eticheta cu cea mai mare probabilitate prezisă de model pentru fiecare text.

8. Salvarea predicțiilor

  La final, am salvat predicțiile modelului într-un fișier CSV pentru a putea analiza rezultatele. Acest fișier conține textele din setul de test și etichetele lor prezise.
