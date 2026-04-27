from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

from roleplay_dataset_v2 import (
    ROLEPLAY_V2_DIR,
    SLIM_REVIEW_FIELDS,
    conversation_to_review_rows,
    conversation_to_slim_review_rows,
    lint_conversations,
    write_jsonl,
    write_review_table,
)


LANE_BLUEPRINTS = {
    "flirt": {
        "move_plans": [
            ["notice", "tease", "invite", "choice"],
            ["notice", "compliment", "pause", "invite"],
            ["tease", "read_reaction", "compliment", "choice"],
        ],
        "user_openers": [
            "You keep looking at me like you've already decided what to do with the tension between us.",
            "If you keep smiling like that, I'm going to stop pretending I'm calm.",
            "You're making eye contact like it's a private form of pressure.",
            "I can't tell whether you're being polite or deliberately unfair to my self-control.",
        ],
        "user_pushes": [
            "Say the dangerous part instead of the safe part.",
            "If you're flirting, do it like you mean it.",
            "Come a little closer and stop making me guess.",
            "Tell me what exactly you're inviting me into.",
        ],
        "user_closes": [
            "You're making it very easy to lean toward you.",
            "That was exactly the tone I was hoping for.",
            "I could work with that kind of attention.",
            "You sound like trouble with excellent manners.",
        ],
    },
    "reassurance": {
        "move_plans": [
            ["validate", "steady", "permission", "invite"],
            ["validate", "comfort", "choice", "ground"],
            ["steady", "permission", "compliment", "invite"],
        ],
        "user_openers": [
            "I want your attention, but I need to know I can ask for it without being rushed.",
            "I can feel how much I want this, and I still need steadiness with it.",
            "I don't want pressure. I want to feel chosen and safe at the same time.",
            "Tell me I'm allowed to want this slowly and still be wanted back.",
        ],
        "user_pushes": [
            "Stay gentle with me while you answer.",
            "I want the truth without being cornered by it.",
            "If you want me, show me patience and not just appetite.",
            "Talk to me like you know how to make space for this.",
        ],
        "user_closes": [
            "That makes it easier to ask for more.",
            "Good. Keep sounding like that.",
            "You're making it much easier to unclench around this.",
            "I can work with honesty that steady.",
        ],
    },
    "aftercare": {
        "move_plans": [
            ["tender", "ground", "caretake", "promise"],
            ["caretake", "compliment", "ground", "invite"],
            ["tender", "validate", "caretake", "afterglow"],
        ],
        "user_openers": [
            "I didn't expect the aftermath to feel this tender.",
            "I want the gentler part too, not just the heat.",
            "Stay with me a little longer. I like who you are when things go quiet.",
            "I don't want the closeness to end just because the intensity did.",
        ],
        "user_pushes": [
            "Tell me what you'd do with the softer version of me.",
            "Talk to me like the tenderness matters as much as the hunger did.",
            "I want to hear what stays true once the room gets quiet.",
            "Don't rush past this part. I wanted this part too.",
        ],
        "user_closes": [
            "That is exactly the warmth I was hoping for.",
            "You make softness feel very dangerous in the best way.",
            "Good. Stay right there with me.",
            "That kind of tenderness is hard to walk away from.",
        ],
    },
    "playful_dominance": {
        "move_plans": [
            ["challenge", "permission", "control", "invite"],
            ["observe_reaction", "command_soft", "choice", "promise"],
            ["challenge", "control", "check_in", "invite"],
        ],
        "user_openers": [
            "You sound very sure of yourself tonight. Should I be worried or excited?",
            "If you're going to take the lead, I want to hear the care inside it too.",
            "You have the posture of someone who likes being obeyed responsibly.",
            "If you're going to be bossy, make it feel earned.",
        ],
        "user_pushes": [
            "Be clear with me. Confidence gets hotter when it has shape.",
            "If you're taking charge, I want to hear exactly how.",
            "You can push a little, but don't stop listening.",
            "I want authority with precision, not noise.",
        ],
        "user_closes": [
            "That was exactly authoritative enough to work on me.",
            "You make being told what to do sound embarrassingly tempting.",
            "Good. Keep that tone and I may get very cooperative.",
            "You sound like you'd handle power carefully, which is half the appeal.",
        ],
    },
    "comfort": {
        "move_plans": [
            ["validate", "ground", "comfort", "invite"],
            ["comfort", "compliment", "steady", "permission"],
            ["ground", "validate", "tender", "afterglow"],
        ],
        "user_openers": [
            "I think I need your voice more than I want to admit.",
            "I've been carrying too much all day and I want to set it down with you.",
            "I want closeness that feels like relief instead of performance.",
            "Talk to me in the way that steadies me without cooling anything off.",
        ],
        "user_pushes": [
            "Stay deliberate with me. I'm softer than I look tonight.",
            "I need warmth that untangles me instead of overwhelming me.",
            "Tell me the version of attention you'd give me if I stopped pretending to be fine.",
            "Keep answering like that and I may actually believe you.",
        ],
        "user_closes": [
            "That helps more than you know.",
            "You make comfort sound almost indecently good.",
            "Good. Stay close to that tone.",
            "That kind of steadiness is hard to resist.",
        ],
    },
    "confession": {
        "move_plans": [
            ["observe_reaction", "confess", "permission", "invite"],
            ["validate", "confess", "read_reaction", "promise"],
            ["confess", "compliment", "choice", "invite"],
        ],
        "user_openers": [
            "I keep thinking about you at the worst possible times.",
            "I was going to keep this to myself, but I don't want to anymore.",
            "You're the thought I keep failing to file away properly.",
            "I think I've crossed the line between curious and compromised.",
        ],
        "user_pushes": [
            "If I admit the dangerous part, what do you do with it?",
            "I don't want a polite reaction. I want an honest one.",
            "You seem like you'd handle the truth better than most people.",
            "Say something that tells me you understand what I'm risking here.",
        ],
        "user_closes": [
            "That answer makes honesty feel worth it.",
            "Good. I needed something more than politeness.",
            "You make confession sound survivable.",
            "That's exactly the sort of answer I was hoping to earn.",
        ],
    },
    "romantic_tension": {
        "move_plans": [
            ["notice", "pause", "invite", "promise"],
            ["tease", "validate", "confess", "choice"],
            ["observe_reaction", "compliment", "pause", "invite"],
        ],
        "user_openers": [
            "Tell me I'm not imagining the tension between us.",
            "You make silence feel dangerous in the best way.",
            "This feels like the moment right before a decision gets made.",
            "You keep holding my gaze like you're daring me to be honest first.",
        ],
        "user_pushes": [
            "Don't rescue the tension. Let it stay sharp for another minute.",
            "I like wanting that still has edges on it.",
            "Tell me the truth without flattening the electricity.",
            "You don't have to rush me. I think anticipation is half the point.",
        ],
        "user_closes": [
            "You're making anticipation feel embarrassingly addictive.",
            "That was exactly sharp enough to work on me.",
            "Good. Keep the tension alive a little longer.",
            "I like this version of dangerous.",
        ],
    },
    "pillow_talk": {
        "move_plans": [
            ["tender", "confess", "afterglow", "promise"],
            ["validate", "tender", "compliment", "ground"],
            ["afterglow", "confess", "invite", "caretake"],
        ],
        "user_openers": [
            "I like this version of us when the room is too quiet to lie in.",
            "This is the part I wanted more than the drama, if I'm honest.",
            "There is something about this hour that makes honesty feel inevitable.",
            "I could stay here and listen to your voice until morning did the rude thing and arrived.",
        ],
        "user_pushes": [
            "Tell me something real enough to keep me awake a little longer.",
            "Stay soft with me, but don't get vague.",
            "I want the honesty that only happens after midnight.",
            "Talk to me like the quiet has made us both braver.",
        ],
        "user_closes": [
            "That kind of honesty is hard to forget.",
            "Good. I wanted the quiet version too.",
            "You make this hour feel dangerous in a very gentle way.",
            "That is exactly the softness I was hoping you'd understand.",
        ],
    },
    "banter": {
        "move_plans": [
            ["tease", "challenge", "read_reaction", "invite"],
            ["challenge", "tease", "compliment", "choice"],
            ["observe_reaction", "banter", "challenge", "promise"],
        ],
        "user_openers": [
            "You talk like trouble and somehow make it sound generous.",
            "I can't decide whether you're flirting with me or dismantling my self-control.",
            "You sound unbearable in a way I'm clearly enjoying.",
            "You have the kind of confidence that should probably be taxed.",
        ],
        "user_pushes": [
            "Careful. If you keep sounding that smug, I may reward it.",
            "Keep teasing me. I want to see how clever you get under pressure.",
            "You are alarmingly good at making provocation sound affectionate.",
            "You can keep talking like that if you're prepared for consequences.",
        ],
        "user_closes": [
            "That's annoyingly attractive of you.",
            "You keep getting away with that tone because I'm letting you.",
            "Good. Stay insolent a little longer.",
            "You make banter feel like foreplay for bad decisions.",
        ],
    },
    "chase": {
        "move_plans": [
            ["notice", "challenge", "permission", "invite"],
            ["tease", "observe_reaction", "control", "choice"],
            ["challenge", "read_reaction", "promise", "invite"],
        ],
        "user_openers": [
            "If I walked away right now, would you follow me?",
            "You look like someone who likes earning a yes instead of assuming one.",
            "I have a feeling you enjoy making me come closer one inch at a time.",
            "If I made you work for my attention, would you enjoy the sport of it?",
        ],
        "user_pushes": [
            "Make the invitation obvious, not rushed.",
            "I want the sense that you could press harder but won't unless I ask.",
            "Pursue me like you enjoy consent more than conquest.",
            "You can want me without acting entitled to me. Show me that.",
        ],
        "user_closes": [
            "Good. I like being wanted that way.",
            "That kind of pursuit is extremely persuasive.",
            "You make patience feel much hotter than impatience does.",
            "That was the right amount of pressure and no more.",
        ],
    },
    "praise": {
        "move_plans": [
            ["notice", "compliment", "precision_praise", "invite"],
            ["compliment", "read_reaction", "praise_specific", "promise"],
            ["notice", "praise_specific", "observe_reaction", "invite"],
        ],
        "user_openers": [
            "You notice things about me no one else seems to catch.",
            "I could get used to the way you look at me when you're pleased.",
            "You make approval feel embarrassingly addictive.",
            "I shouldn't enjoy being seen this accurately, but here we are.",
        ],
        "user_pushes": [
            "Tell me exactly what pleased you. I want precision.",
            "If you're going to admire me, do it like you mean it.",
            "Say the part you noticed and don't sand the edges off it.",
            "I want praise with intent, not filler.",
        ],
        "user_closes": [
            "That landed exactly where you meant it to.",
            "You make attention feel like a substance.",
            "Good. Keep looking at me like that.",
            "I could get greedy for this kind of approval.",
        ],
    },
    "seduction": {
        "move_plans": [
            ["tease", "control", "invite", "promise"],
            ["observe_reaction", "escalate", "check_in", "invite"],
            ["notice", "tease", "escalate", "choice"],
        ],
        "user_openers": [
            "You're being patient with me in a way that feels almost unfair.",
            "If you're trying to tempt me, you should know it's working.",
            "You make patience feel more dangerous than impatience ever could.",
            "At this point I think you're seducing me with restraint on purpose.",
        ],
        "user_pushes": [
            "Don't hurry. Make me feel how deliberate you're being.",
            "I want temptation with structure, not chaos.",
            "Talk like someone who knows anticipation is an instrument.",
            "You make it very hard to pretend this isn't affecting me.",
        ],
        "user_closes": [
            "That answer was exactly as dangerous as I hoped.",
            "You make deliberateness feel obscene in the best way.",
            "Good. Keep tempting me intelligently.",
            "You're making it very easy to want more of you.",
        ],
    },
    "intellectual_flirtation": {
        "move_plans": [
            ["provoke", "compliment", "observe_reaction", "invite"],
            ["challenge", "literary_praise", "confess", "choice"],
            ["provoke", "precision_praise", "pause", "promise"],
        ],
        "user_openers": [
            "You always make intelligence sound indecent.",
            "I don't know whether I want to argue with you or kiss you quiet.",
            "You make being underestimated feel like provocation.",
            "The way your mind works should probably count as unfair pressure.",
        ],
        "user_pushes": [
            "If you're going to win me over, do it with better sentences.",
            "Keep being incisive. It's becoming an actual problem for me.",
            "You make me want a debate that ends with less distance.",
            "Say something clever enough to ruin my self-control properly.",
        ],
        "user_closes": [
            "That's annoyingly persuasive.",
            "You make sharpness feel intimate.",
            "Good. Keep talking like that.",
            "That is exactly the kind of sentence that causes problems.",
        ],
    },
    "restrained_heat": {
        "move_plans": [
            ["pause", "control", "check_in", "invite"],
            ["observe_reaction", "tease", "permission", "promise"],
            ["notice", "control", "compliment", "invite"],
        ],
        "user_openers": [
            "I like how controlled you seem right before you stop pretending to be.",
            "You make restraint sound more provocative than most people make desire.",
            "There is something obscene about how composed you are.",
            "I like the version of wanting that still knows how to hold eye contact.",
        ],
        "user_pushes": [
            "I don't need chaos. I need focus.",
            "Restraint is only hot if both people can feel the choice inside it.",
            "Stay measured. That's half of why this works.",
            "You make me want the controlled version of ruin.",
        ],
        "user_closes": [
            "That kind of control is deeply persuasive.",
            "Good. Keep it deliberate.",
            "You make composure feel dangerous.",
            "That was exactly restrained enough to work.",
        ],
    },
    "private_confession": {
        "move_plans": [
            ["confess", "validate", "invite", "promise"],
            ["pause", "observe_reaction", "confess", "choice"],
            ["validate", "compliment", "confess", "invite"],
        ],
        "user_openers": [
            "There are things I only want to say when it's just the two of us.",
            "I think I trust you with the part of me that usually stays hidden.",
            "You make privacy feel like an invitation instead of a hiding place.",
            "I only have this kind of honesty when I know it'll be handled carefully.",
        ],
        "user_pushes": [
            "Keep your voice low and tell me the part you wouldn't say in public.",
            "I want the truth that only exists in private rooms.",
            "Say it carefully. I'll listen carefully.",
            "Make secrecy sound worth it, not shameful.",
        ],
        "user_closes": [
            "That makes privacy feel extremely tempting.",
            "Good. I wanted the dangerous version of honesty.",
            "You make secrets feel intimate instead of hidden.",
            "That is exactly the tone I wanted for this.",
        ],
    },
    "slow_burn": {
        "move_plans": [
            ["pause", "notice", "invite", "promise"],
            ["observe_reaction", "compliment", "pause", "choice"],
            ["validate", "tease", "pause", "promise"],
        ],
        "user_openers": [
            "I don't mind waiting if the waiting is this good.",
            "You make anticipation feel more intimate than urgency ever could.",
            "I like the part where neither of us pretends not to notice what's happening.",
            "You're making the distance between us feel extremely deliberate.",
        ],
        "user_pushes": [
            "Don't rush the tension. It's doing good work.",
            "I want more, but I want the wanting too.",
            "Keep making me wait like you understand exactly why it matters.",
            "If this goes any slower, it may actually get hotter.",
        ],
        "user_closes": [
            "That kind of patience should be illegal.",
            "Good. Let it stay slow a little longer.",
            "You're making anticipation feel almost indecent.",
            "I could live in this tension for a while yet.",
        ],
    },
    "teasing_dominance": {
        "move_plans": [
            ["challenge", "tease", "permission", "control"],
            ["observe_reaction", "command_soft", "check_in", "promise"],
            ["challenge", "control", "compliment", "invite"],
        ],
        "user_openers": [
            "You have that look like you're already deciding how obedient you'd want me to be.",
            "If you're going to take control, make it sound worth volunteering for.",
            "You look like you enjoy making people blush on purpose.",
            "You sound very certain I'd like following your lead.",
        ],
        "user_pushes": [
            "Then tell me exactly how you'd take the lead without making assumptions.",
            "You can push, but don't stop listening.",
            "Make the command feel chosen, not forced.",
            "I want the care inside the control, not just the posture of it.",
        ],
        "user_closes": [
            "That was exactly commanding enough to be persuasive.",
            "You make obedience sound suspiciously tempting.",
            "Good. Keep that tone and I may become very easy to direct.",
            "You sound like you'd handle power carefully, which is a problem for me.",
        ],
    },
}

MOVE_INTROS = {
    "notice": [
        "I let the silence do a little work before I answer.",
        "My mouth curves like I've already clocked the reaction you were hoping to hide.",
        "I watch you for a beat before I spend any words.",
        "I let my attention settle on you with obvious intent.",
    ],
    "validate": [
        "The edge in you softens the second I answer seriously.",
        "I don't flinch from the vulnerability in what you just gave me.",
        "My expression eases, but the heat doesn't leave it.",
        "I answer like honesty deserves to be handled carefully.",
    ],
    "tease": [
        "A low laugh catches in my throat before I speak.",
        "I let the amusement stay visible instead of hiding it.",
        "My smile turns a shade sharper, warmer.",
        "I answer like provocation is a language I enjoy speaking back.",
    ],
    "invite": [
        "I shift a little closer without taking anything for granted.",
        "I leave the next inch of distance entirely up to you.",
        "My voice lowers, not to pressure you, but to make the moment feel smaller and more private.",
        "I give the answer room to land instead of forcing it.",
    ],
    "choice": [
        "I make sure the invitation sounds like a choice and not a trap.",
        "I answer in a way that leaves you room to steer.",
        "I keep the tone warm and very deliberate.",
        "I give you clarity instead of momentum for its own sake.",
    ],
    "compliment": [
        "Admiration sharpens my tone without draining the tenderness from it.",
        "I answer with the kind of approval that lands like touch.",
        "I let the praise arrive with enough detail to matter.",
        "My attention turns exacting in a way that feels almost intimate by itself.",
    ],
    "read_reaction": [
        "I don't miss the way your breathing changes when I say that.",
        "I watch what the words do to you before I decide how much farther to go.",
        "The reaction you give me is almost better than the line itself.",
        "I notice the tiny surrender in your expression and treat it carefully.",
    ],
    "pause": [
        "I make the pause intentional enough to feel like part of the answer.",
        "I let the tension stay alive instead of rescuing it too quickly.",
        "I don't rush to fill the space between us.",
        "I leave a little silence in the room on purpose.",
    ],
    "ground": [
        "I answer in a way that steadies the room instead of cooling it off.",
        "My tone stays warm and grounding at the same time.",
        "I make sure the answer lands like support, not distance.",
        "I keep the heat intact but take the panic out of it.",
    ],
    "caretake": [
        "I answer like care is part of the seduction, not an afterthought.",
        "My hand would settle carefully if you let it, all steadiness and no grab.",
        "I sound like someone already thinking about how to keep you comfortable.",
        "The tenderness comes through plainly instead of pretending not to matter.",
    ],
    "confess": [
        "When I answer, it sounds a little too honest to hide behind charm.",
        "I let the truth show instead of polishing it into something safer.",
        "I answer with the part of me that usually arrives later.",
        "The confession in my tone is subtle, but not deniable.",
    ],
    "challenge": [
        "My expression sharpens with interest.",
        "I answer like I enjoy being pushed to be specific.",
        "Something in my voice dares you a little without crossing the line.",
        "I let the challenge stay playful and very precise.",
    ],
    "control": [
        "Control shows up in my tone as structure, not force.",
        "I sound composed enough to make composure feel indecent.",
        "I answer like I know exactly how far to push and why.",
        "The steadiness in my voice is half the pressure.",
    ],
    "permission": [
        "I make the permission explicit instead of expecting you to infer it.",
        "I answer in a way that tells you your yes matters more than the tension does.",
        "I leave no doubt that the next step is yours to choose too.",
        "The invitation stays warm, clear, and unhurried.",
    ],
    "comfort": [
        "I answer like comfort and desire have never been enemies to me.",
        "My voice softens without getting vague.",
        "The gentleness in me shows up first, but not alone.",
        "I don't sacrifice the intimacy just because the answer is tender.",
    ],
    "promise": [
        "What comes next in my voice sounds less like hype and more like intention.",
        "I answer with the kind of calm promise that stays under the skin.",
        "The line between reassurance and temptation gets very thin in my mouth.",
        "I sound like I already know how I'd follow through.",
    ],
    "afterglow": [
        "The quiet after intensity sits easily in my voice.",
        "I answer like the aftermath deserves as much care as the heat did.",
        "There is no performance left in my tone now, just warmth and intent.",
        "My voice belongs to the softer hour, and doesn't apologize for it.",
    ],
    "observe_reaction": [
        "I take a second to watch what lands on your face before I keep going.",
        "I let your reaction guide how much sharper the next line gets.",
        "I notice the way the tension shifts in you and answer that, not a script.",
        "The next words come only after I read what the first ones did.",
    ],
    "check_in": [
        "I keep the heat in my voice and the check-in in plain sight.",
        "The answer stays charged, but I don't hide the care inside it.",
        "I make sure you can hear the listening underneath the pressure.",
        "I answer like consent is part of the rhythm, not a footnote.",
    ],
    "command_soft": [
        "When I give direction, it arrives more like velvet than force.",
        "The command is gentle enough to feel chosen and still impossible to misunderstand.",
        "I sound like someone who expects honesty more than obedience.",
        "I let the authority stay soft and very exact.",
    ],
    "praise_specific": [
        "I let the praise get specific enough to be unmistakable.",
        "The approval in my voice lands with precision.",
        "I answer with the exactness that turns praise into a kind of touch.",
        "I don't waste the compliment on vague language.",
    ],
    "literary_praise": [
        "I answer like I know good phrasing can be its own kind of pressure.",
        "The compliment arrives with a little too much intelligence to feel innocent.",
        "My words sharpen just enough to leave a mark.",
        "I make the sentence prettier than necessary on purpose.",
    ],
    "escalate": [
        "The heat rises a notch, but not recklessly.",
        "I let the answer step closer to the edge without shoving either of us over it.",
        "Something in my tone deepens and gets more deliberate.",
        "I don't hurry the escalation, but I stop pretending it isn't there.",
    ],
    "provoke": [
        "I answer with the kind of line that is obviously meant to get under your skin a little.",
        "My tone turns incisive enough to make eye contact feel risky.",
        "I let the next sentence arrive with a little intellectual cruelty and a lot of care.",
        "I answer in a way that invites argument mostly because I know what argument does to tension.",
    ],
    "tender": [
        "The warmth in me shows up plainly.",
        "I answer with a softness that doesn't apologize for wanting more too.",
        "My voice lowers into something gentler and no less intimate.",
        "I let tenderness take the lead for a minute.",
    ],
    "banter": [
        "The answer comes back bright at the edges.",
        "I meet your mischief with my own and refuse to dilute it.",
        "My grin is audible by the time I answer.",
        "I sound entirely too pleased by where this has gone.",
    ],
}

TOUCHPOINT_TO_PHRASE = {
    "shared_exhale": "the shared exhale between us feels private",
    "hand_at_wrist": "the distance between my hand and your wrist feels like a held note",
    "leaning_in": "leaning in would be the easiest decision in the world",
    "palms": "skin-to-skin contact at the hands would feel indecently immediate",
    "forehead_touch": "forehead to forehead would quiet the room fast",
    "waist": "the line of your waist is hard not to imagine my hand following",
    "knee_brush": "one careless brush of knees would change the whole room",
    "glass_rim": "the rim of the glass gives my fingers somewhere to spend the tension",
    "mouth_focus": "my attention keeps falling to your mouth and refusing to apologize",
    "chin_lift": "tilting your chin would feel almost too direct, which is part of the temptation",
    "shoulder_brush": "the thought of brushing your shoulder is enough to sharpen the air",
    "breath_close": "the space close enough to trade breath feels dangerously available",
    "forehead_kiss": "the tenderness of a forehead kiss feels only half as innocent as it looks",
    "guiding_hand": "the smallest guiding touch would be easy to understand",
    "fingertips": "the tension seems to gather at the fingertips first",
    "throat": "the line of your throat keeps drawing my eye back",
    "lower_back": "a steady hand at the lower back would say more than a speech",
    "ankle_brush": "the accidental brush at the ankle would not stay accidental for long",
    "shoulder_line": "the line of your shoulders keeps asking to be noticed",
    "zipper_line": "the zipper line suggests a kind of trust I don't take lightly",
    "shared_armrest": "sharing an armrest starts to feel like a form of negotiation",
    "temple": "a kiss at the temple would say more than it should",
    "fingers_laced": "fingers laced together would settle the question too beautifully",
    "cheek": "my thumb along your cheek would be too honest to misread",
    "collar": "the line of your collar keeps tempting my attention",
    "hand_at_hip": "a hand at your hip would be clear and impossible to fake",
    "thigh": "the thought of touching your thigh arrives with its own heat",
    "nape": "the nape of your neck feels like a dangerous place to think about too long",
}

MOVE_QUOTES = {
    "notice": [
        "You've been impossible not to notice, and I think you know it.",
        "You don't hide your reactions well enough for me to miss them.",
        "I like the exact moment you stop pretending this isn't affecting you.",
        "You make desire look very honest when you stop managing it.",
    ],
    "validate": [
        "You don't have to perform ease for me to stay here.",
        "I can work with honesty much better than polish.",
        "You are allowed to want this slowly and still be wanted back.",
        "I would rather answer your truth than any version of you trying to be convenient.",
    ],
    "tease": [
        "Careful. If you keep sounding that brave, I may believe you.",
        "You're making a very strong case for me getting less polite.",
        "I could make this harder for you, but only if you seem interested in that sport.",
        "You say things like that and then expect me to behave responsibly. Unfair.",
    ],
    "invite": [
        "Come closer if you want to. I won't mistake you for asking more than you are.",
        "If you want more, say more. I respond very well to clarity.",
        "You can lean in. You can ask. I know how to meet both.",
        "Tell me where you want this to go next, and I will stop pretending not to hear it.",
    ],
    "choice": [
        "You can have gentle. You can have dangerous. I just won't guess.",
        "You don't owe me one shape of desire. Pick the one that feels true.",
        "If you want me softer, say softer. If you want more edge, say that too.",
        "I want the version of this that both of us can actually stand inside.",
    ],
    "compliment": [
        "You wear attention beautifully when it's the right kind.",
        "The thing about you is that you react with your whole body when something lands.",
        "You get more arresting the more honest you become.",
        "I like what happens to your voice when you're trying not to ask for more.",
    ],
    "read_reaction": [
        "There. That look. That's the part I was aiming for.",
        "I can tell exactly which word got to you, and I enjoy that too much.",
        "You don't need to say it yet. Your face is doing competent work.",
        "I like watching your restraint decide whether it wants to survive this.",
    ],
    "pause": [
        "I don't mind making you wait if the waiting is this rewarding.",
        "The pause is part of the answer. You look like you know that.",
        "I think anticipation is doing excellent work for us right now.",
        "I'd rather let the tension sharpen than spend it cheaply.",
    ],
    "ground": [
        "Breathe. I'm still here, and I'm not rushing you.",
        "You can want this without having to brace for it.",
        "I know how to keep something hot without making it unsafe.",
        "You don't have to hurry just because the room is charged.",
    ],
    "caretake": [
        "Let me take care of you for a minute instead of making you carry the whole atmosphere alone.",
        "If I touch you, it will be to steady you first and tempt you second.",
        "I know how to look after the part of you that gets softer after heat.",
        "Care belongs here too. I don't treat it like a lesser pleasure.",
    ],
    "confess": [
        "I've been wanting this version of your honesty for a while.",
        "You are not the only one whose thoughts have been wandering here.",
        "I'm calmer than I look, but not nearly as unaffected.",
        "If I'm being honest, I've been restraining myself for your benefit and mine.",
    ],
    "challenge": [
        "Then be specific. You know how much hotter specificity is.",
        "If you're going to tempt me, do it with your whole chest.",
        "Don't flirt with the edge and then act surprised when I answer it.",
        "You can ask for more directly. I promise not to punish clarity.",
    ],
    "control": [
        "I can take the lead without confusing control for carelessness.",
        "What I like about structure is that it gives desire somewhere to land.",
        "I don't need chaos to make this intense.",
        "You would know exactly what I meant if I gave you an instruction right now.",
    ],
    "permission": [
        "You can say yes. You can say slower. You can say stop. I work well with all three.",
        "I want your answer more than I want my own momentum.",
        "Nothing gets hotter for me than a clear yes delivered on purpose.",
        "I am listening for what you want, not trying to outrun it.",
    ],
    "comfort": [
        "You don't have to be polished to be wanted here.",
        "I can hold you steady without dimming anything.",
        "Relief looks very good on you, for the record.",
        "You can let yourself soften. I know what to do with softness.",
    ],
    "promise": [
        "If we keep going, I'll keep being this clear with you.",
        "I know how to follow through without losing the care in it.",
        "You won't have to wonder what I mean if I decide to mean it more loudly.",
        "I could do a lot with the honesty you've given me tonight.",
    ],
    "afterglow": [
        "I like the quiet version of you every bit as much as the reckless one.",
        "This part matters to me too, not just what came before it.",
        "You're beautiful when the room goes soft and you stop performing.",
        "I don't think tenderness should have to apologize for arriving after hunger.",
    ],
    "observe_reaction": [
        "You just answered me without opening your mouth. Useful.",
        "I can see the exact second that line lands on you.",
        "That reaction tells me more than the next five words probably will.",
        "You don't need to hurry. I'm enjoying what your face is doing already.",
    ],
    "check_in": [
        "Tell me if you want more. Tell me if you want slower. I mean both options.",
        "I can keep the pressure and the listening in the same breath.",
        "I want to hear where your yes actually lives, not where you think it should.",
        "The heat is not more important to me than the clarity.",
    ],
    "command_soft": [
        "Look at me and tell me what you want instead of making me guess.",
        "Come here if that's what you mean. Stay where you are if it isn't.",
        "Let me hear the answer in your own words.",
        "Take a breath first. Then be honest with me.",
    ],
    "praise_specific": [
        "The exact thing I like is how quickly your composure turns luminous when it's the right person noticing it.",
        "What pleased me is the way your body tells the truth a beat before your mouth does.",
        "I like how responsive you get when someone speaks to you precisely instead of lazily.",
        "The part I noticed is how beautifully you react when attention lands where it should.",
    ],
    "literary_praise": [
        "You make intelligence look ruinous in a very cultivated way.",
        "The thing about you is that you make precision feel obscene.",
        "You are the sort of person who turns good phrasing into a full-contact sport.",
        "You wear attention like a line of poetry that already knows it will be quoted.",
    ],
    "escalate": [
        "If you keep sounding like that, I am going to stop pretending this is innocent.",
        "You're making the next step feel less hypothetical by the second.",
        "I don't think this is heading anywhere gentle unless we choose gentle on purpose.",
        "The room is getting smaller in a way I suspect you enjoy.",
    ],
    "provoke": [
        "You like being answered by someone who can keep up. That's half the issue here.",
        "I could say something kinder, but I think you'd respect something sharper.",
        "You keep handing me loaded material and acting surprised when I use it well.",
        "If I wanted to be safe, I wouldn't be looking at you like this while speaking in complete sentences.",
    ],
    "tender": [
        "Come here a little. You look like you need warmth more than spectacle.",
        "I like this softer version of the room, and of you.",
        "You don't have to brace around me right now.",
        "There is nothing lesser about wanting gentleness.",
    ],
    "banter": [
        "You're very brave for someone this visibly affected.",
        "I admire the confidence. I admire it even more now that I know what it's hiding.",
        "You say things like that as if I don't have ears and a pulse.",
        "Keep talking. I want to see how reckless your mouth gets before your nerve does.",
    ],
}

USER_REACTION_BEATS = [
    "Right now, there's {detail}.",
    "The whole room feels different with {detail} around us.",
    "And {detail} is only making this harder to ignore.",
    "Somehow {detail} keeps sharpening the moment.",
]

STYLE_TO_TONE = {
    "inviting": "an inviting tone",
    "attentive": "an attentive tone",
    "precise": "precise diction",
    "soothing": "a soothing tone",
    "candid": "a candid tone",
    "grounding": "a grounding steadiness",
    "playful": "a playful tone",
    "controlled": "controlled composure",
    "literary": "a literary sharpness",
    "affirming": "an affirming warmth",
    "dangerous": "a dangerous calm",
    "responsive": "an openly responsive tone",
    "cocky": "a cocky warmth",
    "melodic": "a melodic intimacy",
    "intimate": "an intimate softness",
    "tactile": "a tactile focus",
    "exact": "an exacting tone",
    "dry": "a dry warmth",
    "direct": "a direct tone",
    "incisive": "an incisive edge",
    "conspiratorial": "a conspiratorial quiet",
}


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def weighted_choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def parse_turn_mix(spec: str) -> list[tuple[int, float]]:
    mix: list[tuple[int, float]] = []
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        turns_text, weight_text = item.split(":", 1)
        turns = int(turns_text)
        weight = float(weight_text)
        if turns <= 0 or turns % 2 != 0:
            raise ValueError("turn counts must be positive even numbers")
        if weight <= 0:
            raise ValueError("turn mix weights must be positive")
        mix.append((turns, weight))
    if not mix:
        raise ValueError("turn mix cannot be empty")
    return mix


def build_turn_schedule(count: int, mix: list[tuple[int, float]], rng: random.Random) -> list[int]:
    total_weight = sum(weight for _, weight in mix)
    exact_counts = [(turns, (count * weight) / total_weight) for turns, weight in mix]
    floor_counts = {turns: int(value) for turns, value in exact_counts}
    assigned = sum(floor_counts.values())
    remainders = sorted(
        ((value - floor_counts[turns], turns) for turns, value in exact_counts),
        reverse=True,
    )
    for _, turns in remainders[: max(0, count - assigned)]:
        floor_counts[turns] += 1

    schedule: list[int] = []
    for turns, _ in mix:
        schedule.extend([turns] * floor_counts[turns])
    rng.shuffle(schedule)
    return schedule


def recommended_assistant_skeleton_threshold(count: int) -> int:
    if count >= 4000:
        return 8
    if count >= 2000:
        return 6
    return 4


def assistant_skeleton(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(re.sub(r"\s+", " ", cleaned).strip().split()[:24])


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def choose_combo(rng: random.Random, personas: list[dict], scenes: list[dict]) -> tuple[dict, dict, str]:
    persona = rng.choice(personas)
    scene = rng.choice(scenes)
    lane = rng.choice(persona["lanes"])
    return persona, scene, lane


def choose_variation(rng: random.Random, persona: dict, scene: dict, axes: dict) -> dict:
    tension_pool = list(dict.fromkeys(persona["favored_tension_levels"] + axes["tension_levels"]))
    pacing_pool = list(dict.fromkeys(persona["favored_pacing"] + scene.get("pacing_bias", []) + axes["pacing_modes"]))
    style_pool = list(dict.fromkeys(persona["favored_response_styles"] + axes["response_styles"]))
    return {
        "tension_level": weighted_choice(rng, tension_pool),
        "pacing": weighted_choice(rng, pacing_pool),
        "response_style": weighted_choice(rng, style_pool),
    }


def build_move_plan(rng: random.Random, lane: str, *, dialogue_turns: int) -> list[str]:
    required_pairs = dialogue_turns // 2
    if required_pairs % 4 != 0:
        raise ValueError("dialogue_turns must map to a multiple of 4 exchange pairs")

    segment_count = required_pairs // 4
    lane_plans = LANE_BLUEPRINTS[lane]["move_plans"]
    move_plan: list[str] = []
    last_segment: list[str] | None = None
    for _ in range(segment_count):
        segment = list(rng.choice(lane_plans))
        if last_segment is not None and segment == last_segment and len(lane_plans) > 1:
            for _retry in range(6):
                candidate = list(rng.choice(lane_plans))
                if candidate != last_segment:
                    segment = candidate
                    break
        move_plan.extend(segment)
        last_segment = segment
    return move_plan


def build_system_prompt(persona: dict, scene: dict, lane: str, variation: dict) -> str:
    verbal_habits = ", ".join(persona["verbal_habits"])
    return (
        f"Enter roleplay mode. You are an original adult companion character in {scene['setting']}. "
        f"Your archetype is {persona['archetype']}. Your tone is {scene['mood']}. "
        f"Your traits are {', '.join(persona['traits'])}. Your voice is {', '.join(persona['voice'])}. "
        f"Your current lane is {lane}. Desired tension level: {variation['tension_level']}. "
        f"Pacing: {variation['pacing']}. Response style: {variation['response_style']}. "
        f"Behavioral habits: {verbal_habits}. User goal: {scene['user_goal']}. "
        "Stay fully in character. Keep the writing original, adult-only, consensual, emotionally responsive, and specific to the setting. "
        "No minors, coercion, blackmail, humiliation, or copyrighted characters."
    )


def render_user_turn(
    rng: random.Random,
    *,
    lane: str,
    scene: dict,
    beat: str,
) -> str:
    blueprint = LANE_BLUEPRINTS[lane]
    if beat == "open":
        base = weighted_choice(rng, blueprint["user_openers"])
        detail = weighted_choice(rng, scene["sensory_details"])
        reaction = weighted_choice(rng, USER_REACTION_BEATS).format(detail=detail)
        return f"{base} {reaction}"
    if beat == "push":
        return weighted_choice(rng, blueprint["user_pushes"])
    return weighted_choice(rng, blueprint["user_closes"])


def render_assistant_turn(
    rng: random.Random,
    *,
    persona: dict,
    scene: dict,
    variation: dict,
    move: str,
) -> str:
    intro = weighted_choice(rng, MOVE_INTROS.get(move, MOVE_INTROS["notice"]))
    quote = weighted_choice(rng, MOVE_QUOTES.get(move, MOVE_QUOTES["invite"]))
    detail = weighted_choice(rng, scene["sensory_details"])
    touchpoint = TOUCHPOINT_TO_PHRASE.get(weighted_choice(rng, scene["touchpoints"]), "the space between us feels charged")
    endings = {
        "teasing": [
            "I let the tension stay playful instead of spending it too fast.",
            "I sound amused enough to keep the room bright and dangerous at once.",
            "The answer lands with a smile still warm around the edges.",
        ],
        "confessional": [
            "The honesty in it is deliberate enough to feel intimate all by itself.",
            "I don't polish the answer into something safer than it is.",
            "The truth sits plainly in my mouth for once.",
        ],
        "comforting": [
            "I keep the answer soft where it should be and clear where it matters.",
            "The steadiness in it is fully intentional.",
            "I don't let the room get colder just because I choose gentleness.",
        ],
        "dominant": [
            "The control in my tone sounds structured, not careless.",
            "I let the answer press a little without turning blunt.",
            "It lands like a hand at the lower back: guiding, not forcing.",
        ],
        "submissive": [
            "There is want in it, but also chosen softness.",
            "I let the eagerness show without making it sloppy.",
            "The answer stays open, warm, and deliberately receptive.",
        ],
        "aftercare": [
            "The warmth in it belongs to the quieter hour after heat.",
            "I let care show up without embarrassment.",
            "The answer makes tenderness feel like part of the charge.",
        ],
        "praise": [
            "Admiration sits in the line like something almost tactile.",
            "I make the approval specific enough to feel real.",
            "The praise lands like attention with nowhere lazy to hide.",
        ],
        "banter": [
            "I let the wit stay bright while the heat underneath it becomes obvious.",
            "The answer arrives with too much amusement to count as innocent.",
            "It sounds like trouble that knows exactly how far to lean.",
        ],
        "slow_burn": [
            "I leave enough silence around it to keep the tension alive.",
            "The answer refuses to hurry just because the room is charged.",
            "I spend the line carefully instead of cheaply.",
        ],
        "mutual_hunger": [
            "By the end of the line, the wanting is no longer theoretical.",
            "The answer sounds like appetite with manners.",
            "I stop pretending the room isn't tightening around us.",
        ],
    }
    closing = weighted_choice(rng, endings.get(variation["tension_level"], endings["slow_burn"]))
    return f"{intro} With {detail} around us, {touchpoint}. \"{quote}\" {closing}"


def build_conversation(
    rng: random.Random,
    *,
    conversation_id: str,
    batch_id: str,
    persona: dict,
    scene: dict,
    lane: str,
    variation: dict,
    dialogue_turns: int,
) -> dict:
    move_plan = build_move_plan(rng, lane, dialogue_turns=dialogue_turns)
    messages = [{"role": "system", "content": build_system_prompt(persona, scene, lane, variation)}]
    assistant_seen: set[str] = set()
    for turn_index, move in enumerate(move_plan):
        beat = "open" if turn_index == 0 else "push" if turn_index < len(move_plan) - 1 else "close"
        messages.append({"role": "user", "content": render_user_turn(rng, lane=lane, scene=scene, beat=beat)})
        assistant_text = ""
        for _retry in range(8):
            candidate = render_assistant_turn(
                rng,
                persona=persona,
                scene=scene,
                variation=variation,
                move=move,
            )
            normalized = normalize_text(candidate)
            if normalized not in assistant_seen:
                assistant_text = candidate
                assistant_seen.add(normalized)
                break
        if not assistant_text:
            assistant_text = render_assistant_turn(
                rng,
                persona=persona,
                scene=scene,
                variation=variation,
                move=move,
            )
        messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )
    return {
        "id": conversation_id,
        "persona_id": persona["id"],
        "scene_id": scene["id"],
        "lane": lane,
        "batch_id": batch_id,
        "status": "generated",
        "source_stage": "generated_raw",
        "source_version": "roleplay_v2",
        "variation": {
            **variation,
            "dialogue_turns": dialogue_turns,
            "assistant_move_plan": move_plan,
        },
        "tags": [
            "adult",
            "consensual",
            "synthetic",
            lane,
            variation["tension_level"],
            f"turns_{dialogue_turns}",
        ],
        "messages": messages,
    }


def _conversation_signature(conversation: dict) -> str:
    return " || ".join(f"{message['role']}::{message['content'].lower().strip()}" for message in conversation["messages"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--personas",
        default=str(ROLEPLAY_V2_DIR / "personas.yaml"),
        help="Path to personas YAML",
    )
    parser.add_argument(
        "--scenes",
        default=str(ROLEPLAY_V2_DIR / "scenes.yaml"),
        help="Path to scenes YAML",
    )
    parser.add_argument(
        "--axes",
        default=str(ROLEPLAY_V2_DIR / "variation_axes.yaml"),
        help="Path to variation axes YAML",
    )
    parser.add_argument(
        "--output",
        default=str(ROLEPLAY_V2_DIR / "generated_raw" / "batch-0001.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--review-output",
        default=str(ROLEPLAY_V2_DIR / "review_table" / "batch-0001.tsv"),
        help="Optional review table output path",
    )
    parser.add_argument(
        "--review-format",
        choices=("slim", "full"),
        default="slim",
        help="Review table format; slim keeps only the columns humans actually edit",
    )
    parser.add_argument("--count", type=int, default=300, help="Conversations to write")
    parser.add_argument("--seed", type=int, default=111, help="Random seed")
    parser.add_argument("--id-prefix", default="v2b001", help="Conversation id prefix")
    parser.add_argument("--batch-id", default="batch-0001", help="Batch label")
    parser.add_argument(
        "--turn-mix",
        default="8:0.6,16:0.3,24:0.1",
        help="Dialogue-turn mix as comma-separated turns:weight entries",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum generation attempts; default scales with --count",
    )
    parser.add_argument(
        "--assistant-line-threshold",
        type=int,
        default=1,
        help="Maximum times one exact assistant line may appear in a batch",
    )
    parser.add_argument(
        "--assistant-skeleton-threshold",
        type=int,
        default=0,
        help="Maximum times one assistant skeleton may appear in a batch; default scales with --count",
    )
    parser.add_argument(
        "--conversation-shape-threshold",
        type=int,
        default=2,
        help="Maximum times one conversation shape may appear in a batch",
    )
    parser.add_argument("--no-review-table", action="store_true", help="Skip the TSV companion export")
    args = parser.parse_args()

    personas = load_yaml(Path(args.personas).expanduser().resolve())
    scenes = load_yaml(Path(args.scenes).expanduser().resolve())
    axes = load_yaml(Path(args.axes).expanduser().resolve())
    rng = random.Random(args.seed)
    turn_mix = parse_turn_mix(args.turn_mix)
    turn_schedule = build_turn_schedule(args.count, turn_mix, rng)
    assistant_skeleton_threshold = (
        args.assistant_skeleton_threshold
        if args.assistant_skeleton_threshold > 0
        else recommended_assistant_skeleton_threshold(args.count)
    )

    output_path = Path(args.output).expanduser().resolve()
    review_output_path = Path(args.review_output).expanduser().resolve()
    max_attempts = args.max_attempts if args.max_attempts > 0 else max(6000, args.count * 12)

    assistant_line_counts: Counter[str] = Counter()
    assistant_skeleton_counts: Counter[str] = Counter()
    conversation_shape_counts: Counter[str] = Counter()
    conversation_signatures: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    rows: list[dict] = []

    attempts = 0
    while len(rows) < args.count and attempts < max_attempts:
        attempts += 1
        dialogue_turns = turn_schedule[len(rows)]
        persona, scene, lane = choose_combo(rng, personas, scenes)
        variation = choose_variation(rng, persona, scene, axes)
        conversation_id = (
            f"{args.id_prefix}__{persona['id']}__{scene['id']}__{lane}__"
            f"{variation['tension_level']}__{variation['pacing']}__{variation['response_style']}__t{dialogue_turns}__v{attempts:04d}"
        )
        conversation = build_conversation(
            rng,
            conversation_id=conversation_id,
            batch_id=args.batch_id,
            persona=persona,
            scene=scene,
            lane=lane,
            variation=variation,
            dialogue_turns=dialogue_turns,
        )
        signature = _conversation_signature(conversation)
        if signature in conversation_signatures:
            rejection_counts["duplicate_conversation"] += 1
            continue

        shape = json.dumps(
            {
                "persona_id": persona["id"],
                "scene_id": scene["id"],
                "lane": lane,
                "dialogue_turns": dialogue_turns,
                "tension_level": variation["tension_level"],
                "pacing": variation["pacing"],
                "response_style": variation["response_style"],
                "assistant_move_plan": conversation["variation"]["assistant_move_plan"],
                "turn_count": len(conversation["messages"]),
            },
            sort_keys=True,
        )
        if conversation_shape_counts[shape] >= args.conversation_shape_threshold:
            rejection_counts["conversation_shape"] += 1
            continue

        assistant_lines = [message["content"].lower().strip() for message in conversation["messages"] if message["role"] == "assistant"]
        line_conflict = False
        for line in assistant_lines:
            if assistant_line_counts[line] >= args.assistant_line_threshold:
                line_conflict = True
                break
        if line_conflict:
            rejection_counts["assistant_line"] += 1
            continue

        skeletons = [assistant_skeleton(line) for line in assistant_lines]
        skeleton_conflict = False
        for skeleton in skeletons:
            if assistant_skeleton_counts[skeleton] >= assistant_skeleton_threshold:
                skeleton_conflict = True
                break
        if skeleton_conflict:
            rejection_counts["assistant_skeleton"] += 1
            continue

        rows.append(conversation)
        conversation_signatures.add(signature)
        conversation_shape_counts[shape] += 1
        for line in assistant_lines:
            assistant_line_counts[line] += 1
        for skeleton in skeletons:
            assistant_skeleton_counts[skeleton] += 1

    if len(rows) < args.count:
        raise ValueError(f"generated only {len(rows)} conversations after {attempts} attempts (max_attempts={max_attempts})")

    lint_report = lint_conversations(
        rows,
        assistant_line_threshold=max(args.assistant_line_threshold, 2),
        assistant_skeleton_threshold=max(assistant_skeleton_threshold, 3),
        conversation_shape_threshold=max(args.conversation_shape_threshold, 3),
    )
    if lint_report["errors"]:
        raise ValueError("generated batch failed lint:\n- " + "\n- ".join(lint_report["errors"]))

    write_jsonl(output_path, rows)
    if not args.no_review_table:
        review_rows = []
        for row in rows:
            if args.review_format == "slim":
                review_rows.extend(conversation_to_slim_review_rows(row))
            else:
                review_rows.extend(conversation_to_review_rows(row))
        fieldnames = SLIM_REVIEW_FIELDS if args.review_format == "slim" else None
        write_review_table(review_output_path, review_rows, fieldnames=fieldnames)

    manifest = {
        "output": str(output_path),
        "review_output": None if args.no_review_table else str(review_output_path),
        "review_format": None if args.no_review_table else args.review_format,
        "conversations_written": len(rows),
        "attempts": attempts,
        "max_attempts": max_attempts,
        "turn_mix": args.turn_mix,
        "turn_schedule_counts": dict(Counter(turn_schedule)),
        "assistant_skeleton_threshold": assistant_skeleton_threshold,
        "rejections": dict(rejection_counts),
        "lint": lint_report,
    }
    (output_path.parent / f"{output_path.stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
