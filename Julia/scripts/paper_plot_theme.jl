module PaperPlotTheme

using CairoMakie

const NEWCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
const REGULAR = joinpath(NEWCM_DIR, "NewCM10-Regular.otf")
const ITALIC = joinpath(NEWCM_DIR, "NewCM10-Italic.otf")
const BOLD = joinpath(NEWCM_DIR, "NewCM10-Bold.otf")
const BOLDITALIC = joinpath(NEWCM_DIR, "NewCM10-BoldItalic.otf")
const MATH = joinpath(NEWCM_DIR, "NewCMMath-Regular.otf")

function setup_mathfonts!()
    mte_id = Base.PkgId(Base.UUID("0a4f8689-d25c-4efe-a92b-7142dfc1aa53"), "MathTeXEngine")
    mte = get(Base.loaded_modules, mte_id, nothing)
    mte === nothing && return
    try
        mte.set_texfont_family!(regular = REGULAR, italic = ITALIC, bold = BOLD,
            bolditalic = BOLDITALIC, math = MATH)
    catch err
        @warn "Unable to register New Computer Modern math fonts" exception = (err, catch_backtrace())
    end
end

function with_theme(f)
    setup_mathfonts!()
    CairoMakie.with_theme(CairoMakie.theme_latexfonts();
        fonts = (; regular = REGULAR, italic = ITALIC, bold = BOLD, bolditalic = BOLDITALIC)) do
        f()
    end
end

end
