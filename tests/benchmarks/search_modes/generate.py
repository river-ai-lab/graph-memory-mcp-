#!/usr/bin/env python3
"""Deterministic generator for the structural retrieval benchmark (no LLM).

Writes dataset.json: core facts with entities and typed edges and ~40 questions
with gold fact ids. build() also returns 10k noise facts (same templates,
disjoint entities); run.py rebuilds them from the seed instead of a 4 MB file.
Copied from the Scenario 1 harness (bench/retrieval-structural).
Run: python tests/benchmarks/search_modes/generate.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SEED = 1729
NOISE_MAX = 10_000
OUT = Path(__file__).with_name("dataset.json")

FIRST = """Nora Ivo Lena Omar Tariq Mia Jonas Ada Felix Rosa Kai Elif Hugo Sana Pavel
Ines Luca Yara Bruno Dana Emil Freya Gustav Hana Igor Jana Karl Lina Milo Nadia Oskar
Petra Rafael Selma Tomas Ula Viktor Wanda Xenia Yusuf Zora Anton Bela Cyra Dario Edda
Filip Greta Henrik Ilse""".split()
LAST = """Lindqvist Petrov Moreau Haddad Okafor Varga Brandt Silva Novak Keller Tanaka
Duarte Berg Kowalski Romero Fischer Sato Ivanova Laine Costa Weber Horvat Nilsen Rossi
Yilmaz Dvorak Meyer Popescu Sorensen Marino Hansen Kovac Lund Ortega Becker Holm Szabo
Engel Ricci Falk""".split()
TEAMS = """Falcon Heron Otter Lynx Badger Kestrel Marten Osprey Wren Ibis Puffin Stoat
Raven Egret Viper Bison Crane Dingo Gecko Hare Jackal Koala Lemur Mole Newt Orca Panda
Quail Robin Swift Tapir Urial Vole Wolf Yak Zebu Adder Bream Cobra Drake""".split()
MACHINE_KINDS = [
    "lathe",
    "press",
    "mill",
    "grinder",
    "welder",
    "router",
    "kiln",
    "loom",
]
LOCS = """Harbor Quarry Summit Meadow Delta Ridge Canyon Prairie Lagoon Fjord Mesa Bluff
Grove Marsh Glacier Dune Basin Cove Plateau Strait Tundra Oasis Reef Steppe Valley
Butte Inlet Moor Atoll Cliff""".split()
PROJECTS = """Orion Vega Lyra Draco Hydra Cetus Pavo Indus Norma Ara Crux Mensa Volans
Musca Lepus Corvus Aquila Cygnus Pyxis Tucana Fornax Carina Puppis Vela Dorado Pictor
Columba Sextans Antlia Caelum""".split()
CITIES = """Tallinn Porto Graz Ghent Turku Lyon Brno Split Cork Bergen Malmo Pecs Riga
Kaunas Basel Bari Leeds Varna Ulm Nantes""".split()
HOBBIES = """chess climbing pottery fencing birdwatching sailing archery origami juggling
beekeeping""".split()
ATTRS = [
    ("budget", ["40k EUR", "55k EUR", "70k EUR", "85k EUR", "30k EUR", "95k EUR"]),
    ("deadline", ["March", "May", "July", "September", "November", "January"]),
    ("status", ["planning", "prototype", "pilot", "rollout", "paused", "audit"]),
]

# Sentence templates. Noise reuses the same templates with other entities.
T_MEMBER = "{p} belongs to team {t}."
T_LEAD = "Team {t} is led by {p}."
T_OPER = "{p} runs the {m}."
T_MLOC = "The {m} sits in the {l} depot."
T_PLOC = "Project {j} is hosted at the {l} site."
T_OWNS = "Team {t} owns project {j}."
T_ATTR = "Project {j} {a} is {v}."
T_BORN = "{p} was born in {c}."
T_HOBBY = "{p} spends weekends on {h}."


class Builder:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.facts: list[dict] = []
        self.entities: dict[str, dict] = {}
        self.edges: list[dict] = []

    def ent(self, kind: str, name: str) -> str:
        eid = f"{self.prefix}e:{kind}:{name}"
        self.entities.setdefault(
            eid, {"id": eid, "kind": kind, "text": f"{name} ({kind})"}
        )
        return eid

    def fact(self, kind: str, text: str, ents: list[str], **extra) -> str:
        fid = f"{self.prefix}f{len(self.facts):05d}"
        self.facts.append(
            {
                "id": fid,
                "kind": kind,
                "text": text,
                "entities": ents,
                "status": "active",
                **extra,
            }
        )
        for e in ents:
            self.edges.append({"from": fid, "to": e, "type": "MENTIONS"})
        return fid

    def rel(self, a: str, b: str, rtype: str) -> None:
        self.edges.append({"from": a, "to": b, "type": rtype})


def world(rng: random.Random, people, teams, machines, locs, projects, n_members=3):
    """Assign structure: teams with members+lead, machines, projects."""
    s = {
        "member_of": {},
        "lead": {},
        "operates": {},
        "mloc": {},
        "ploc": {},
        "owns": {},
    }
    pool = list(people)
    rng.shuffle(pool)
    for t in teams:
        lead = pool.pop()
        s["lead"][t] = lead
        for _ in range(n_members):
            s["member_of"][pool.pop()] = t
    leads = list(s["lead"].values())
    for p, m in zip(leads + pool, machines):
        s["operates"][p] = m
    for m in machines:
        s["mloc"][m] = rng.choice(locs)
    for j in projects:
        s["ploc"][j] = rng.choice(locs)
        s["owns"][j] = rng.choice(teams)
    s["rest"] = pool
    return s


def emit_world(b: Builder, rng, s, cities, hobbies, attr_chains: int) -> dict:
    idx: dict = {
        "member": {},
        "lead": {},
        "oper": {},
        "mloc": {},
        "ploc": {},
        "owns": {},
        "attr": {},
        "born": {},
        "hobby": {},
    }
    E = b.ent
    for p, t in s["member_of"].items():
        f = b.fact("member", T_MEMBER.format(p=p, t=t), [E("person", p), E("team", t)])
        b.rel(E("person", p), E("team", t), "MEMBER_OF")
        idx["member"][p] = f
    for t, p in s["lead"].items():
        f = b.fact("lead", T_LEAD.format(t=t, p=p), [E("team", t), E("person", p)])
        b.rel(E("person", p), E("team", t), "LEADS")
        idx["lead"][t] = f
    for p, m in s["operates"].items():
        f = b.fact("oper", T_OPER.format(p=p, m=m), [E("person", p), E("machine", m)])
        b.rel(E("person", p), E("machine", m), "OPERATES")
        idx["oper"][p] = f
    for m, loc in s["mloc"].items():
        f = b.fact(
            "mloc", T_MLOC.format(m=m, l=loc), [E("machine", m), E("location", loc)]
        )
        b.rel(E("machine", m), E("location", loc), "LOCATED_IN")
        idx["mloc"][m] = f
    for j, loc in s["ploc"].items():
        f = b.fact(
            "ploc", T_PLOC.format(j=j, l=loc), [E("project", j), E("location", loc)]
        )
        b.rel(E("project", j), E("location", loc), "LOCATED_IN")
        idx["ploc"][j] = f
    for j, t in s["owns"].items():
        f = b.fact("owns", T_OWNS.format(t=t, j=j), [E("team", t), E("project", j)])
        b.rel(E("team", t), E("project", j), "OWNS")
        idx["owns"][j] = f
    projects = list(s["ploc"])
    for i in range(attr_chains):
        j = projects[i % len(projects)]
        a, values = ATTRS[i % len(ATTRS)]
        n = 2 + (i % 2) + 1  # 3 or 4 versions = 2 or 3 updates
        vals = rng.sample(values, n)
        chain = []
        for v in vals:
            f = b.fact("attr", T_ATTR.format(j=j, a=a, v=v), [E("project", j)], value=v)
            if chain:
                b.rel(f, chain[-1], "FOLLOWS_FROM")
            chain.append(f)
        for f in chain[:-1]:
            b.facts[int(f[len(b.prefix) + 1 :])]["status"] = "outdated"
        idx["attr"][(j, a)] = chain
    people = list(s["member_of"]) + list(s["lead"].values()) + s["rest"]
    for p in people:
        f = b.fact("born", T_BORN.format(p=p, c=rng.choice(cities)), [E("person", p)])
        idx["born"][p] = f
    return idx


def build() -> dict:
    """Return the full dataset (core, noise, questions). Deterministic."""
    rng = random.Random(SEED)
    names = [f"{a} {b}" for a in FIRST for b in LAST]
    rng.shuffle(names)
    core_people = names[:26]
    noise_people = names[26:]
    mk = [
        f"{k} {c}-{n}"
        for k in MACHINE_KINDS
        for c in "ABCDEFGHKLMNPRSTVXYZ"
        for n in range(1, 10)
    ]
    rng.shuffle(mk)

    # ---- core world
    b = Builder("")
    teams, locs, projects = TEAMS[:6], LOCS[:4], PROJECTS[:8]
    s = world(rng, core_people, teams, mk[:12], locs, projects)
    idx = emit_world(b, rng, s, CITIES, HOBBIES, attr_chains=6)
    facts = {f["id"]: f for f in b.facts}

    def has_overlap(q: str, fid: str) -> bool:
        stop = {
            "the",
            "a",
            "of",
            "is",
            "in",
            "at",
            "to",
            "on",
            "by",
            "which",
            "what",
            "who",
            "where",
            "does",
            "that",
            "s",
            "its",
            "for",
            "and",
            "now",
            "with",
        }

        def tok(x: str) -> set[str]:
            words = x.replace("'s", " ").split()
            return {w.strip(".,?'").lower() for w in words} - stop

        return bool(tok(q) & tok(facts[fid]["text"]))

    qs: list[dict] = []
    # 3-hop: member -> team -> lead -> machine
    members = sorted(s["member_of"])
    for p in members[:8]:
        t = s["member_of"][p]
        lead = s["lead"][t]
        if lead not in s["operates"]:
            continue
        ans = idx["oper"][lead]
        q = f"Which equipment does the person heading {p}'s crew work with?"
        qs.append(
            {
                "type": "chain3",
                "q": q,
                "gold": [idx["member"][p], idx["lead"][t], ans],
                "answer_fact": ans,
            }
        )
    # 2-hop: person -> machine -> location
    ops = sorted(s["operates"])
    for p in ops[:8]:
        m = s["operates"][p]
        ans = idx["mloc"][m]
        q = f"In which facility is the gear that {p} handles kept?"
        qs.append(
            {
                "type": "chain2",
                "q": q,
                "gold": [idx["oper"][p], ans],
                "answer_fact": ans,
            }
        )
    # aggregation: team members; projects per location
    for t in teams:
        gold = [idx["member"][p] for p, tt in sorted(s["member_of"].items()) if tt == t]
        qs.append(
            {
                "type": "aggregate",
                "q": f"List everyone who belongs to team {t}.",
                "gold": gold,
            }
        )
    for loc in locs:
        gold = [idx["ploc"][j] for j in projects if s["ploc"][j] == loc]
        if len(gold) >= 2:
            qs.append(
                {
                    "type": "aggregate",
                    "q": f"Name all projects hosted at the {loc} site.",
                    "gold": gold,
                }
            )
    # current value after 2-3 updates
    for (j, a), chain in idx["attr"].items():
        qs.append(
            {
                "type": "current",
                "q": f"What is the current {a} of project {j}?",
                "gold": [chain[-1]],
                "stale": chain[:-1],
            }
        )
    # single-fact controls
    for p in sorted(idx["born"])[:10]:
        qs.append(
            {"type": "control", "q": f"Where was {p} born?", "gold": [idx["born"][p]]}
        )

    for q in qs:
        if "answer_fact" in q:
            assert not has_overlap(q["q"], q["answer_fact"]), (
                q["q"],
                facts[q["answer_fact"]]["text"],
            )
        q["id"] = f"q{len([x for x in qs if 'id' in x]):02d}"

    # ---- noise: same templates, disjoint entities, built in shards of worlds
    nb = Builder("n")
    nrng = random.Random(SEED + 1)
    nteams = TEAMS[6:]
    nlocs = LOCS[4:]
    nproj = [f"{a}-{n}" for a in PROJECTS[8:] for n in range(1, 40)]
    nrng.shuffle(nproj)
    pi = mi = ji = 0
    while len(nb.facts) < NOISE_MAX:
        ts = nrng.sample(nteams, 6)
        ts = [f"{t}-{len(nb.facts) // 50}" for t in ts]
        ppl = [noise_people[(pi + i) % len(noise_people)] for i in range(26)]
        pi += 26
        ppl = [
            f"{p} {chr(65 + (pi // len(noise_people)) % 26)}."
            if pi >= len(noise_people)
            else p
            for p in ppl
        ]
        ms = [mk[12 + (mi + i) % (len(mk) - 12)] for i in range(12)]
        mi += 12
        js = [nproj[(ji + i) % len(nproj)] for i in range(8)]
        ji += 8
        sw = world(nrng, ppl, ts, ms, nrng.sample(nlocs, 4), js)
        emit_world(nb, nrng, sw, CITIES, HOBBIES, attr_chains=6)
    nb.facts = nb.facts[:NOISE_MAX]
    keep = {f["id"] for f in nb.facts}
    nb.edges = [
        e
        for e in nb.edges
        if (not e["from"].startswith("nf") or e["from"] in keep)
        and (not e["to"].startswith("nf") or e["to"] in keep)
    ]
    # every noise entity that no kept fact mentions is dropped
    used = {e for f in nb.facts for e in f["entities"]}
    nb.edges = [
        e
        for e in nb.edges
        if (e["from"] in keep or e["from"] in used)
        and (e["to"] in keep or e["to"] in used)
    ]
    out = {
        "seed": SEED,
        "core": {
            "facts": b.facts,
            "entities": list(b.entities.values()),
            "edges": b.edges,
        },
        "noise": {
            "facts": nb.facts,
            "entities": [v for k, v in nb.entities.items() if k in used],
            "edges": nb.edges,
        },
        "questions": qs,
    }
    return out


def main() -> None:
    """Write core facts and questions. Noise is rebuilt from the seed at run time."""
    out = build()
    noise = out.pop("noise")
    out["noise_summary"] = {
        "facts": len(noise["facts"]),
        "entities": len(noise["entities"]),
        "edges": len(noise["edges"]),
    }
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: len(v) for k, v in out["core"].items()}), out["noise_summary"])


if __name__ == "__main__":
    main()
