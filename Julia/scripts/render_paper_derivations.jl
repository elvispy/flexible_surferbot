#!/usr/bin/env julia
#
# Render test/test_paper_derivations.jl as a document.
#
# The test file is written in literate style: lines beginning with "# " are
# prose in Markdown, everything else is Julia. This script splits the two, wraps
# the code in fenced blocks, and hands the result to pandoc.
#
#     julia Julia/scripts/render_paper_derivations.jl
#
# Requires pandoc and a LaTeX installation. Writes Julia/output/pdf/paper_derivations.pdf.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SRC = joinpath(ROOT, "test", "test_paper_derivations.jl")
const OUTDIR = joinpath(ROOT, "output", "pdf")
const PDF = joinpath(OUTDIR, "paper_derivations.pdf")

function to_markdown(path)
    chunks = String[]
    code = String[]
    function flush_code!()
        while !isempty(code) && isempty(strip(first(code)))
            popfirst!(code)
        end
        while !isempty(code) && isempty(strip(last(code)))
            pop!(code)
        end
        isempty(code) || push!(chunks, "\n```julia\n" * join(code, "\n") * "\n```\n")
        empty!(code)
    end
    for line in eachline(path)
        if startswith(line, "# ")
            flush_code!()
            push!(chunks, line[3:end])
        elseif line == "#"
            flush_code!()
            push!(chunks, "")
        else
            push!(code, line)
        end
    end
    flush_code!()
    join(chunks, "\n")
end

mkpath(OUTDIR)
mktempdir() do tmp
    md = joinpath(tmp, "paper_derivations.md")
    write(md, to_markdown(SRC))
    run(`pandoc $md -o $PDF --pdf-engine=pdflatex
         -V geometry:margin=2.5cm -V fontsize=10pt --highlight-style=tango`)
end
println("wrote ", PDF)
