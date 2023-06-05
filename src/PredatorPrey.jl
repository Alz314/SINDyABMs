module PredatorPrey

    using Agents, Random, InteractiveDynamics, GLMakie
    @agent Sheep GridAgent{2} begin
        energy::Float64
        reproduction_prob::Float64
        Δenergy::Float64
    end

    @agent Wolf GridAgent{2} begin
        energy::Float64
        reproduction_prob::Float64
        Δenergy::Float64
    end

    function initialize_model(;
            n_sheep = 100,
            n_wolves = 50,
            dims = (20, 20),
            regrowth_time = 30,
            Δenergy_sheep = 4,
            Δenergy_wolf = 20,
            sheep_reproduce = 0.04,
            wolf_reproduce = 0.05,
            seed = 23182,
        )

        rng = MersenneTwister()
        space = GridSpace(dims, periodic = true)
        # Model properties contain the grass as two arrays: whether it is fully grown
        # and the time to regrow. Also have static parameter `regrowth_time`.
        # Notice how the properties are a `NamedTuple` to ensure type stability.
        properties = (
            fully_grown = falses(dims),
            countdown = zeros(Int, dims),
            regrowth_time = regrowth_time,
        )
        model = ABM(Union{Sheep, Wolf}, space;
            properties, rng, scheduler = Schedulers.randomly, warn = false
        )
        # Add agents
        for _ in 1:n_sheep
            energy = rand(model.rng, 1:(Δenergy_sheep*2)) - 1
            add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep)
        end
        for _ in 1:n_wolves
            energy = rand(model.rng, 1:(Δenergy_wolf*2)) - 1
            add_agent!(Wolf, model, energy, wolf_reproduce, Δenergy_wolf)
        end
        # Add grass with random initial growth
        for p in positions(model)
            fully_grown = rand(model.rng, Bool)
            countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
            model.countdown[p...] = countdown
            model.fully_grown[p...] = fully_grown
        end
        return model
    end

    function sheepwolf_step!(sheep::Sheep, model)
        randomwalk!(sheep, model)
        sheep.energy -= 1
        if sheep.energy < 0
            remove_agent!(sheep, model)
            return
        end
        eat!(sheep, model)
        if rand(model.rng) ≤ sheep.reproduction_prob
            reproduce!(sheep, model)
        end
    end
    
    function sheepwolf_step!(wolf::Wolf, model)
        randomwalk!(wolf, model; ifempty=false)
        wolf.energy -= 1
        if wolf.energy < 0
            remove_agent!(wolf, model)
            return
        end
        # If there is any sheep on this grid cell, it's dinner time!
        dinner = first_sheep_in_position(wolf.pos, model)
        !isnothing(dinner) && eat!(wolf, dinner, model)
        if rand(model.rng) ≤ wolf.reproduction_prob
            reproduce!(wolf, model)
        end
    end
    
    function first_sheep_in_position(pos, model)
        ids = ids_in_position(pos, model)
        j = findfirst(id -> model[id] isa Sheep, ids)
        isnothing(j) ? nothing : model[ids[j]]::Sheep
    end

    function eat!(sheep::Sheep, model)
        if model.fully_grown[sheep.pos...]
            sheep.energy += sheep.Δenergy
            model.fully_grown[sheep.pos...] = false
        end
        return
    end
    
    function eat!(wolf::Wolf, sheep::Sheep, model)
        remove_agent!(sheep, model)
        wolf.energy += wolf.Δenergy
        return
    end
    
    function reproduce!(agent::A, model) where {A}
        agent.energy /= 2
        id = nextid(model)
        offspring = A(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy)
        add_agent_pos!(offspring, model)
        return
    end

    function grass_step!(model)
        @inbounds for p in positions(model) # we don't have to enable bound checking
            if !(model.fully_grown[p...])
                if model.countdown[p...] ≤ 0
                    model.fully_grown[p...] = true
                    model.countdown[p...] = model.regrowth_time
                else
                    model.countdown[p...] -= 1
                end
            end
        end
    end

    function simulate_system(sheep_pop, steps, sheep_density = 5120 / 180^2, wolf_density = 720 / 180^2)
        space_dim = sqrt(sheep_pop / sheep_density)
        space_dim = round(Int, space_dim)
        stable_params = (;
            n_sheep = sheep_pop,
            n_wolves = round(Int, wolf_density * space_dim^2),
            dims = (space_dim, space_dim),
            Δenergy_sheep = 5,
            sheep_reproduce = 0.31,
            wolf_reproduce = 0.06,
            Δenergy_wolf = 30,
            seed = 71758,
        )

        sheep(a) = a isa Sheep
        wolf(a) = a isa Wolf
        count_grass(model) = count(model.fully_grown)
        adata = [(sheep, count), (wolf, count)]
        mdata = [count_grass]

        sheepwolfgrass = initialize_model(;stable_params...)
        adf, mdf = run!(sheepwolfgrass, sheepwolf_step!, grass_step!, steps; adata, mdata)
        data = hcat(mdf.count_grass, adf.count_sheep, adf.count_wolf)
        return data
    end

    function simulate_grass_sheep_system(pop, steps; density = 0.125)
        space_dim = sqrt(pop/density)
        space_dim = Int(round(space_dim))
        stable_params = (;
            n_sheep = pop,
            n_wolves = 0,
            dims = (space_dim, space_dim),
            Δenergy_sheep = 5,
            sheep_reproduce = 0.31,
            wolf_reproduce = 0.16,
            Δenergy_wolf = 15,
            seed = 71758,
        )

        sheep(a) = a isa Sheep
        wolf(a) = a isa Wolf
        count_grass(model) = count(model.fully_grown)
        adata = [(sheep, count), (wolf, count)]
        mdata = [count_grass]

        sheepwolfgrass = initialize_model(;stable_params...)
        adf, mdf = run!(sheepwolfgrass, sheepwolf_step!, grass_step!, steps; adata, mdata)
        data = hcat(adf.count_sheep, mdf.count_grass)
        return data
    end

    function make_video_grass_sheep_system(pop, steps; density = 0.125)
        space_dim = sqrt(pop/density)
        space_dim = Int(round(space_dim))
        stable_params = (;
            n_sheep = pop,
            n_wolves = 0,
            dims = (space_dim, space_dim),
            Δenergy_sheep = 5,
            sheep_reproduce = 0.31,
            wolf_reproduce = 0.16,
            Δenergy_wolf = 15,
            seed = 71758,
        )
        sheepwolfgrass = initialize_model(;stable_params...)

        offset(a) = a isa Sheep ? (-0.1, -0.1*rand()) : (+0.1, +0.1*rand())
        ashape(a) = a isa Sheep ? :circle : :utriangle
        acolor(a) = a isa Sheep ? RGBAf(1.0, 1.0, 1.0, 0.8) : RGBAf(0.2, 0.2, 0.3, 0.8)

        grasscolor(model) = model.countdown ./ model.regrowth_time
        heatkwargs = (colormap = [:brown, :green], colorrange = (0, 1))
        plotkwargs = (;
            ac = acolor,
            as = 25,
            am = ashape,
            offset,
            scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
            heatarray = grasscolor,
            heatkwargs = heatkwargs,
        )

        abmvideo(
            "sheepgrass.mp4",
            sheepwolfgrass,
            sheepwolf_step!,
            grass_step!;
            frames = 100,
            framerate = 8,
            title = "Sheep Grass",
            plotkwargs...,
        )
    end
end