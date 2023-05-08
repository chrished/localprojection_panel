# Simulate local projection TWFE with shocks
using DataFrames
using FixedEffectModels
using Distributions
using StatsPlots
using Query 

function simuldata(I, T, ρ, β0, Dϵ, Ds, y0)
    # draw error 
    ϵ = [rand(Dϵ[i]) for t = 1:T, i=1:I]
    # draw shocks 
    s = [rand(Ds[t, i]) for t=1:T, i=1:I]
    # initialize y and fill
    y = zeros(T,I)
    y[1,:] .= β0 .* s[1,:] .+ ρ .* y0 .+  ϵ[1,:] 
    for t = 2:T
        y[t,:] .= β0 .* s[t,:] .+ ρ .* y[t-1,:] .+  ϵ[t,:]     
    end
    # DataFrame for regression 
    df = DataFrame(y = vcat([y[:,i] for i =1:I]...),
                   s = vcat([s[:,i] for i =1:I]...),
                   ID = vcat([ones(Int, T)*i for i = 1:I]...),
                   t=vcat([1:T for i = 1:I]...))
    return df
end

function simul_LP_TWFE(I, T, ρ, β0, Dϵ, Ds, y0) 
    df = simuldata(I, T, ρ, β0, Dϵ, Ds, y0) 
    df.ym1 = collect(lag(df.y, 1))
    df[df.t.==1, :ym1] .= missing  
    df.yp1 = collect(lead(df.y, 1))
    df[df.t.==maximum(df.t), :yp1] .= missing  
    df.yp2 = collect(lead(df.y, 2))
    df[df.t.==maximum(df.t), :yp2] .= missing  
    df[df.t.==maximum(df.t)-1, :yp2] .= missing  
        
    ## Regression 
    reg0 = reg(df, @formula(y ~ s + ym1 +  fe(ID) + fe(t)))
    reg1 = reg(df, @formula(yp1 ~ s + ym1 +  fe(ID) + fe(t)))
    reg2 = reg(df, @formula(yp2 ~ s + ym1 +  fe(ID) + fe(t)))
    reg0_trend = reg(df, @formula(y ~ s + ym1 +  fe(ID)&t + fe(t)))
    reg1_trend = reg(df, @formula(yp1 ~ s + ym1 +  fe(ID)&t + fe(t)))
    reg2_trend = reg(df, @formula(yp2 ~ s + ym1 +  fe(ID)&t + fe(t)))

    βLP = [coef(reg0)[1], coef(reg1)[1], coef(reg2)[1]]
    βLP_trend = [coef(reg0_trend)[1], coef(reg1_trend)[1], coef(reg2_trend)[1]]
    return βLP,βLP_trend
end 


## Simulation settings 
## Case 1
S = 10_000 # number of Monte Carlo repititions
T=100 # time periods 
I=2 # number of units
ρ = zeros(I) .+ 0.5
β0 = ones(I) 
σ = 0.1
Dϵ = [Normal(0, σ) for i=1:I]
σs = 0.1 
Ds = [Normal(0, σs) for t=1:T, i =1:I]
y0 = zeros(I) # Initial condition 


## run simulation 
βLP_sim = zeros(S, 3)
βLP_trend_sim = zeros(S, 3)
for s = 1:S
    βLP, βLP_trend = simul_LP_TWFE(I, T, ρ, β0, Dϵ, Ds, y0)
    βLP_sim[s,:] .= βLP
    βLP_trend_sim[s,:] .= βLP_trend 
end

βLP_avg = mean(βLP_sim, dims=1)[:]
βtrue = [β0[1], β0[1]*ρ[1], β0[1]*ρ[1]^2]

# density plot 
density(βLP_sim[:,1])
vline!([βtrue[1]], label="true value", color=:red)
vline!([βLP_avg[1]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh0case1.pdf")

density(βLP_sim[:,2])
vline!([βtrue[2]], label="true value", color=:red)
vline!([βLP_avg[2]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh1case1.pdf")

density(βLP_sim[:,3])
vline!([βtrue[3]], label="true value", color=:red)
vline!([βLP_avg[3]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh2case1.pdf")


## Case 2
# Now allow for period specific mean of treatment effects 
# treatment grows linearly to 1 over sample  
Ds_trend = [Normal(t/T, σs) for t=1:T, i =1:I]

for s = 1:S
    βLP, βLP_trend = simul_LP_TWFE(I, T, ρ, β0, Dϵ, Ds_trend, y0)
    βLP_sim[s,:] .= βLP
    βLP_trend_sim[s,:] .= βLP_trend 
end


βLP_avg = mean(βLP_sim, dims=1)[:]
βtrue = [β0[1], β0[1]*ρ[1], β0[1]*ρ[1]^2]

# density plot 
density(βLP_sim[:,1])
vline!([βtrue[1]], label="true value", color=:red)
vline!([βLP_avg[1]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh0case2.pdf")

density(βLP_sim[:,2])
vline!([βtrue[2]], label="true value", color=:red)
vline!([βLP_avg[2]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh1case2.pdf")

density(βLP_sim[:,3])
vline!([βtrue[3]], label="true value", color=:red)
vline!([βLP_avg[3]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh2case2.pdf")

## Case 3
# Heterogenous Treatment effects, but otherwise no complications
β0_het = range(0.5,1, length=I)

for s = 1:S
    βLP, βLP_trend = simul_LP_TWFE(I, T, ρ, β0_het, Dϵ, Ds, y0)
    βLP_sim[s,:] .= βLP
    βLP_trend_sim[s,:] .= βLP_trend 
end

βLP_avg = mean(βLP_sim, dims=1)[:]
βtrue_avg = mean(hcat(collect(β0_het), β0_het.*ρ, β0_het.*ρ.^2), dims=1)[:]

# density plot 
density(βLP_sim[:,1])
vline!([βtrue_avg[1]], label="average of true value", color=:red)
vline!([βLP_avg[1]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh0case3.pdf")

density(βLP_sim[:,2])
vline!([βtrue_avg[2]], label="average of true value", color=:red)
vline!([βLP_avg[2]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh1case3.pdf")

density(βLP_sim[:,3])
vline!([βtrue_avg[3]], label="average of true value", color=:red)
vline!([βLP_avg[3]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh2case3.pdf")


## Case 4 a)
# group specific treatment trend 
trendi = range(0,1, length=I)
Ds_hettrend = [Normal(t/T * trendi[i], σs) for t=1:T, i =1:I]
for s = 1:S
    βLP, βLP_trend = simul_LP_TWFE(I, T, ρ, β0, Dϵ, Ds_hettrend, y0)
    βLP_sim[s,:] .= βLP
    βLP_trend_sim[s,:] .= βLP_trend 
end

βLP_avg = mean(βLP_sim, dims=1)[:]
βtrue_avg = mean(hcat(collect(β0), β0.*ρ, β0.*ρ.^2), dims=1)[:]
βLP_trend_avg = mean(βLP_trend_sim, dims=1)[:]

# density plot 
density(βLP_sim[:,1])
vline!([βtrue_avg[1]], label="true value", color=:red)
vline!([βLP_avg[1]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh0case4a.pdf")

density(βLP_sim[:,2])
vline!([βtrue_avg[2]], label="true value", color=:red)
vline!([βLP_avg[2]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh1case4a.pdf")

density(βLP_sim[:,3])
vline!([βtrue_avg[3]], label="true value", color=:red)
vline!([βLP_avg[3]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh2case4a.pdf")

# density plot 
density(βLP_trend_sim[:,1])
vline!([βtrue_avg[1]], label="true value", color=:red)
vline!([βLP_trend_avg[1]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh0case4a_trend.pdf")

density(βLP_trend_sim[:,2])
vline!([βtrue_avg[2]], label="true value", color=:red)
vline!([βLP_trend_avg[2]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh1case4a_trend.pdf")

density(βLP_trend_sim[:,3])
vline!([βtrue_avg[3]], label="true value", color=:red)
vline!([βLP_trend_avg[3]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh2case4a_trend.pdf")

## Case 4 b)
# group specific treatment trend 
trendi = range(0,1, length=I)
Ds_hettrend = [Normal(t/T * trendi[i], σs) for t=1:T, i =1:I]

for s = 1:S
    βLP, βLP_trend = simul_LP_TWFE(I, T, ρ, β0_het, Dϵ, Ds_hettrend, y0)
    βLP_sim[s,:] .= βLP
    βLP_trend_sim[s,:] .= βLP_trend 
end
βLP_avg = mean(βLP_sim, dims=1)[:]
βtrue_avg = mean(hcat(collect(β0_het), β0_het.*ρ, β0_het.*ρ.^2), dims=1)[:]
βLP_trend_avg = mean(βLP_trend_sim, dims=1)[:]

# density plot 
density(βLP_sim[:,1])
vline!([βtrue_avg[1]], label="average of true value", color=:red)
vline!([βLP_avg[1]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh0case4b.pdf")

density(βLP_sim[:,2])
vline!([βtrue_avg[2]], label="average of true value", color=:red)
vline!([βLP_avg[2]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh1case4b.pdf")

density(βLP_sim[:,3])
vline!([βtrue_avg[3]], label="average of true value", color=:red)
vline!([βLP_avg[3]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh2case4b.pdf")

density(βLP_trend_sim[:,1])
vline!([βtrue_avg[1]], label="average of true value", color=:red)
vline!([βLP_trend_avg[1]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh0case4b_trend.pdf")

density(βLP_trend_sim[:,2])
vline!([βtrue_avg[2]], label="average of true value", color=:red)
vline!([βLP_trend_avg[2]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh1case4b_trend.pdf")

density(βLP_trend_sim[:,3])
vline!([βtrue_avg[3]], label="average of true value", color=:red)
vline!([βLP_trend_avg[3]], label="average estimate", color=:black, linestyle=:dash)
savefig("./img/bh2case4b_trend.pdf")
