MILKWISE TOUCH — tablet-style simple cow checks
===============================================

Mac: double-click "Start MilkWise.command" (right-click > Open if macOS blocks it).
Windows: double-click "Start MilkWise.bat".
Linux: run "bash start-milkwise.sh".

BACKUP LAUNCH FOR MAC (if the app refuses to open):
  1. Open Terminal.
  2. Type "cd " (include a space), then drag this folder into Terminal.
  3. Press Return, then type: python3 bootstrap.py

The first launch needs Python 3.10+ and internet to install dependencies into
.milkwise_env INSIDE this folder. No VS Code or global Python setup is changed.
A browser window opens to the local app. Keep the terminal window open while using it.

HOW TO USE THE APP
------------------
1. CHECK: Choose heat (THI) and cow breed group. Optionally add measured feed,
   body, or blood readings. Tap CHECK COW.
2. RESULT: The entire screen changes to GREEN, YELLOW, or RED only if you also
   entered an actual measured milk yield. If you didn't, the result is neutral BLUE.
   This is a production deviation check, NOT a cow health diagnosis.
3. SENSORS: Compare the historical prediction error with fewer/more measurements.
4. DATA: Switch from the hackathon example to your own farm CSV/XLSX. New records
   wait for review, and they are only added after explicit approval.

HISTORY STORAGE
---------------
Approved records only persist for the current browser session. In "Farm data",
click "Save updated history" to keep them. Reupload that exported CSV next time.

DATA LIMITATIONS
----------------
The organizer's physiology workbook has an unresolved cow-ID shift relative to
milk and blood workbooks. Exact-ID joins yield 675 PROVISIONAL matched rows from
45 cow IDs. These data are for a hackathon demonstration, not clinical deployment.

Prediction error ranges are derived from held-out cows and are not guaranteed
individual intervals. A RED screen means unusually LOW observed milk yield
relative to this model's estimate; it is NOT a veterinary diagnosis. Yellow
can also indicate an unusually HIGH result needing a records check.

Milk composition predictions include fat, protein, lactose, solids-not-fat,
salt, pH and somatic cell count. These are not complete nutritional results.

Study dataset: Ahamed (2026), Mendeley Data, https://doi.org/10.17632/954f6g36sb.2
Dataset license: CC BY 4.0.
