/////////////
// Imports //
/////////////
#import "@preview/adaptable-pset:0.2.0": *
#import "@preview/physica:0.9.8": *
#import "@preview/unify:0.8.1": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages)

/////////////////
// Maths Setup //
/////////////////

// upright vectors
#let vectorboldupright(a) = vb($upright(#a)$)
#let vbu = vectorboldupright
#let vectorunitupright(a) = vu($upright(#a)$)
#let vuu = vectorunitupright
#let vectorarrowupright(a) = va($upright(#a)$)
#let vau = vectorarrowupright

// automatically use square brackets for vectors and matricies
#set math.vec(delim: "[")
#set math.mat(delim: "[")
#let vecrowOld = vecrow
#let vecrow = vecrowOld.with(delim: "[")

// custom units
#add-unit("decibelIsotropic", "dBi", "upright(\"dBi\")", space: true)
#add-unit("decibelWatt", "dBW", "upright(\"dBW\")", space: true)

////////////////////
// Document Setup //
////////////////////

// assignment info
#show: homework.with(
    title: "HW04",
    author: "Vai Srivastava",
    collaborators: [],
    course-id: "ENAE 601: Astrodynamics",
    instructor: "Dr. Healy",
    semester: "Fall 2026",
    due-time: datetime(
        // due date
        year: 2026,
        month: 10,
        day: 21,

        // due time
        hour: 23,
        minute: 59,
        second: 00,
    ).display(
        "[month repr:long] [day padding:zero], [year] at [hour repr:24]:[minute padding:zero]:[second padding:zero]",
    ),

    // (defaults to A4)
    paper-size: "us-letter",
)

// document settings
#set text(font: "New Computer Modern", size: 10pt)
#set enum(numbering: "a)")

// problem headings
#let probOld = prob
#let prob = prob.with(color: black)

////////////////////////////
// The Assignment Itself: //
// Problems and Solutions //
////////////////////////////

#prob(title: [Curtis 2.45])[
    Relative to an earth-centered, nonrottating frame, the position and velocity vectors of a spacecraft are $vbu(r)_0 = 6320 vuu(i) + 7750 vuu(k)$ (#unit("km")) and $vbu(v)_0 = 11 vuu(j)$ (#unit("km/s")).
    Use the $Delta theta$ Lagrance coefficient method.

    + Find the position vector #qty(10, "min") later.
    <hwk:p01a>

    + Calculate the change in true anomaly over the #qty(10, "min") time span.
    <hwk:p01b>
] <hwk:p01>

+ Answer
<hwk:s01a>

+ Answer
<hwk:s01b>

#pagebreak(weak: true)

#prob(title: [Curtis 7.7])[
    A space station is in an earth orbit with a #qty(90, "min") period. At $t = 0$, a satellite has the following position and velocity components relative to a CW frame attached to the space station: $delta vbu(r) = 1 vuu(i)$ (#unit("km")) and $vbu(v)_0 = 10 vuu(j)$ (#unit("m/s")). How far is the satellite from the space station #qty(15, "min") later?
] <hwk:p02>

+ Answer
<hwk:s02>

#pagebreak(weak: true)

#prob(title: [Curtis 7.8])[
    Spacecraft $A$ and $B$ are in the same circular earth orbit with a period of #qty(2, "h"). $B$ is #qty(6, "km") ahead of $A$. At $t = 0$, $B$ applies an in-track $Delta v$ (retrofire) of #qty(3, "m/s"). Using a CW frame attached to $A$, determine the distance between $A$ and $B$ at $t = qty(30, "min")$ and the velocity of $B$ relative to $A$ at that instant.
] <hwk:p03>

+ Answer
<hwk:s03>

#pagebreak(weak: true)

#prob(title: [Curtis 7.10])[
    At time $t - 0$, a particle is at the origin of a CW frame with a relative velocity $delta vbu(v)_0 = v vuu(j)$. What will be the relative speed of the particle after a time equal to one-half the orbital period of the CW frame?
] <hwk:p04>

+ Answer
<hwk:s04>

#pagebreak(weak: true)

//----------------------------
// NOTE: ignore below for now
//----------------------------

#prob(title: [Orbit Propagation with STM])[
    + Instructions
    <hwk:p05a>

    + Instructions
    <hwk:p05b>
] <hwk:p05>

+ Answer
<hwk:s05a>

+ Answer
<hwk:s05b>

+ Answer
<hwk:s05c>

#codly(header: [./src/s05c.py])
#raw(read("../src/s05c.py"), block: true, lang: "python") <code:s05c.py>

#pagebreak(weak: true)

#prob(title: [Lambert Problem])[
    Instructions

    + Instructions
    <hwk:p06a>

    + Instructions
    <hwk:p06b>

    + Instructions
    <hwk:p06c>

    + Instructions
    <hwk:p06d>
] <hwk:p06>

+ Answer
<hwk:s06a>

+ Answer
<hwk:s06b>

+ Answer
<hwk:s06c>

#codly(header: [./src/s06c.py])
#raw(read("../src/s06c.py"), block: true, lang: "python") <code:s06c.py>

+ Answer
<hwk:s06d>

#codly(header: [./src/s06d.py])
#raw(read("../src/s06d.py"), block: true, lang: "python") <code:s06d.py>

#pagebreak(weak: true)

== Code

#codly(header: [./src/index.py])
#raw(read("../src/index.py"), block: true, lang: "python") <code:index.py>
