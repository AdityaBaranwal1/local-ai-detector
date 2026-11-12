import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from sanitize import sanitize
FILL = " and that was pretty much the whole story about how the week went for me at work this time around."
CASES = [  # (text, expect) expect: "drop" | "keep" | ("strip", must_not_contain, must_contain)
 ("I was quoted $30 to $40 an hour for the repair, which felt high" + FILL, "keep"),
 ("The study found a significant effect ($p < 0.05$) across both cohorts" + FILL, "drop"),
 ("Location: Austin | Remote: Yes | Willing to relocate: No. Looking for backend roles" + FILL, "keep"),
 ("| Name | Role |\n|---|---|\n| Sam | Lead |\nThat table sums up the team" + FILL, "drop"),
 ("Reach me at jane.doe@example.com if you have questions" + FILL, "drop"),
 ("Call me at (512) 555-0199 any time this week" + FILL, "drop"),
 ("See https://example.com/page for the details" + FILL, "drop"),
 ("I live at 1234 Maple Street now" + FILL, "drop"),
 ("The server is at 192.168.1.10 on the LAN" + FILL, "drop"),
 ("Open C:\\Users\\bob\\Documents and copy the file" + FILL, "drop"),
 ("Hi [Hiring Manager Name], I am applying for the role" + FILL, "drop"),
 ("I used to be a [Native] speaker of nothing in particular, honestly" + FILL, "keep"),
 ("Use users.sort() to order them" + FILL, "drop"),
 ("Step one → step two → done" + FILL, "drop"),
 ("Let me know what you think about the plan for Friday" + FILL, "keep"),
 ("It was 45 minutes each way, so I quit after 2010 and moved on" + FILL, "keep"),
 ("We raised 15% more in Q3 2024 than in Q2, about $1.2M total" + FILL, "keep"),
 ("1. First, update the resume.\n2. Then email the recruiter.\n3. Finally follow up" + FILL, ("strip", "1. ", "update the resume")),
 ("**Bold claim** here and `inline code` there, then more prose" + FILL, ("strip", "**", "Bold claim")),
 ("---\nA separator above this line and a [1] reference" + FILL, ("strip", "[1]", "separator")),
 ("ok so honestly idk lol, the interview went fine i think" + FILL, "keep"),
]
bad = 0
for t, want in CASES:
    out = sanitize(t)
    if want == "drop": ok = out is None
    elif want == "keep": ok = out is not None
    else: ok = out is not None and want[1] not in out and want[2] in out
    bad += not ok
    print("ok  " if ok else "FAIL", want if isinstance(want, str) else "strip", "|", t[:60].replace("\n", " "))
print("failures:", bad)
