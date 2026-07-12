using Test
using SparseArrays
using Surferbot

@testset "assemble_flexible_system" begin
    params = FlexibleParams()
    derived = derive_params(params)
    system = assemble_flexible_system(params)

    NP = derived.Nx * derived.Nz
    nb_contact = count(derived.x_contact)
    expected_size = 2 * NP + nb_contact

    @test issparse(system.A)
    @test size(system.A) == (expected_size, expected_size)
    @test length(system.b) == expected_size
    @test system.derived.Nx == derived.Nx
    @test system.derived.Nz == derived.Nz
    @test count(system.derived.x_contact) == nb_contact
end
