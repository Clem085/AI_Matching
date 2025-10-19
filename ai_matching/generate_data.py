
import random
from pathlib import Path
import pandas as pd

random.seed(7)

# Allow matching across all US states.
STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY",
]
# Mirror the real intake form choices for license type and availability.
LICENSES = [
    "Social Worker",
    "Counselor",
    "Marriage and Family Therapist",
    "Psychologist",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def _format_time(minutes: int) -> str:
    hour = minutes // 60
    minute = minutes % 60
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12}:{minute:02d} {suffix}"

# 30-minute slots from 8:00 AM through 6:00 PM inclusive.
TIME_SLOTS = [_format_time(m) for m in range(8*60, 18*60 + 1, 30)]

FIRST_NAMES = [
    "Alex","Sam","Jordan","Taylor","Morgan","Casey","Riley","Avery","Jamie","Cameron",
    "Drew","Quinn","Hayden","Harper","Rowan","Peyton","Elliot","Skyler","Dakota","Emerson",
    "Finley","Reese","Logan","Kendall",
]
LAST_NAMES = [
    "Lee","Kim","Patel","Garcia","Nguyen","Hernandez","Chen","Brown","Davis","Wilson",
    "Moore","Clark","Lopez","Martinez","Rivera","Cooper","Brooks","Bennett","Reed","Parker",
    "Ramirez","Bailey","Jenkins","Gonzalez",
]

_NAME_POOL = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
random.shuffle(_NAME_POOL)
_name_iter = iter(_NAME_POOL)

def random_name():
    """Return a globally unique name for this synthetic dataset run."""
    try:
        return next(_name_iter)
    except StopIteration:
        raise ValueError("Ran out of unique names; expand FIRST_NAMES/LAST_NAMES.")

def random_email(name):
    base = name.lower().replace(' ', '.')
    dom = random.choice(['example.com','sample.org','mail.net'])
    return f"{base}@{dom}"

def random_availability(k=8):
    universe = [(d, t) for d in DAYS for t in TIME_SLOTS]
    k = min(k, len(universe))
    picks = random.sample(universe, k=k)
    return ', '.join([f"{d} {t}" for d, t in sorted(picks, key=lambda x: (DAYS.index(x[0]), TIME_SLOTS.index(x[1])))])

def generate_supervisors(n=120):
    rows = []
    for i in range(n):
        name = random_name()
        rows.append({
            'Timestamp': '',
            'Email Address': random_email(name),
            'Name': name,
            'State': ', '.join(random.sample(STATES, k=random.randint(2,3))) if random.random() < 0.08 else random.choice(STATES),
            'Who can you supervise?': ', '.join(random.sample(LICENSES, k=random.randint(1,3))),
            'Availability': random_availability(k=random.randint(10,18)),
            'Capacity': 0 if random.random() < 0.05 else random.randint(2,6),
        })
    return pd.DataFrame(rows)

def generate_associates(n=200):
    rows = []
    for i in range(n):
        name = random_name()
        rows.append({
            'Timestamp': '',
            'Email Address': random_email(name),
            'Name': name,
            'State': ', '.join(random.sample(STATES, k=random.randint(2,3))) if random.random() < 0.03 else random.choice(STATES),
            'License Type': random.choice(LICENSES),
            'Availability': random_availability(k=random.randint(6,12)),
        })
    return pd.DataFrame(rows)

def main(out_dir='.'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sup = generate_supervisors()
    assoc = generate_associates()
    sup.to_csv(out/'Supervision_Supervisors_SYNTH.csv', index=False)
    assoc.to_csv(out/'Supervision_Associates_SYNTH.csv', index=False)
    print('Wrote synthetic supervisors & associates.')

if __name__ == '__main__':
    main()
