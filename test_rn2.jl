using MKL
using LinearAlgebra, BenchmarkTools,Distributions,ITensors
#MKL_NUM_THREADS=8

function qr_haar(N)
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A = rand(Normal(0, 1), (N,N))
    B = rand(Normal(0, 1), (N,N))
    Z=(A+B*im)/(2^0.5)
    Q=qr(Z).Q
    R=qr(Z).R
    Rdiag=diagm([R[i,i]/abs(R[i,i]) for i in 1:1:N])
    Q=Q*Rdiag*Q
    
    
    return Q
    
end


function hilbert_schmidt(M1, M2)
    """Hilbert-Schmidt-Product of two matrices M1, M2"""
    return (tr((M1')*M2))
end


function haar_gate(s1::Index{Int64},s2::Index{Int64})
    H=qr_haar(4)

    Sx=[0 1.0;1.0 0]
    Sy=[0.0 -im;im 0]
    Sz=[1.0 0;0 -1]
    Id=[1.0 0;0 1]

    pauli_mat=[Sx,Sy,Sz,Id]
    label=["Sx","Sy","Sz","Id"]


    hj=ITensor()

    #result=zeros(ComplexF64, 4, 4)
    for i in 1:1:length(pauli_mat)
        for j in 1:1:length(pauli_mat)

            a_ij = 0.25 * hilbert_schmidt(kron(pauli_mat[i], pauli_mat[j]), H)
            if a_ij != 0.0
                    #println(a_ij," ",label[i]," x ",label[j])
                    #flush(stdout)
                #result=result+a_ij * kron(pauli_mat[i], pauli_mat[j])
                x=1
                if(i<4 && j<4)
                    x=4
                elseif (i<4)
                     x=2
                elseif (j<4)
                    x=2
                end

                hj = hj+x*a_ij* op(label[i],s1) * op(label[j],s2) 
            end
        end
    end
    
    return hj
end




function time_evolve_layer(psi::MPS,start::Int64,prob_measure::Float64)
    
    sites=siteinds(psi)
    N=length(sites)
    
    
    # Make gates (1,2),(2,3),(3,4),...
    gates = ITensor[]
    sites=siteinds(psi)
    N=length(sites)
    
    for j=start:2:N-1
        
        s1 = sites[j]
        s2 = sites[j+1]
        #A = randomITensor(s1', s1)  
        hg=haar_gate(s1,s2)
        push!(gates,hg)

        #@show A

    end


    psi = apply(gates, psi;cutoff=1E-16 )
    
    
        
        
       

    for j in start:1:N+1-start
    	if (prob_measure>rand())
	    	proj_dn = expect(psi,"ProjDn";site_range=(j):(j))
		if(proj_dn>rand())
		
		    orthogonalize!(psi,j)
		    newA = op("ProjDn",sites[j]) * psi[j]
		    noprime!(newA)
		    psi[j]=newA


		else
		    orthogonalize!(psi,j)
		    newA = op("ProjUp",sites[j]) * psi[j]
		    noprime!(newA)
		    psi[j]=newA
	
		end
		
	psi=(1/norm(psi))*psi
	
	end

    end






    
    
    return psi
end


let

N = 24
cutoff = 1E-18
b=div(N,2)



sites = siteinds("S=1/2",N;conserve_qns=false)

prob_measure=parse(Float64,ARGS[1])
num_of_trials=parse(Int64,ARGS[2])
data_S02=[]
data_SvN2=[]
filename="simdata_jl_2_"*string(prob_measure)*".txt"
f = open(filename, "w")

 for trial in 1:1:num_of_trials
    
    psi = productMPS(sites, n -> isodd(n) ? "Up" : "Dn")
    

    println("HEllo")
    
    for time in 1:1:48
        if(time%2==1)
            
            psi=time_evolve_layer(psi,1,prob_measure)
            #println("at even time ",time," norm = ",norm(psi))
    

        else

            psi=time_evolve_layer(psi,2,prob_measure)
            #println("at even time ",time," norm = ",norm(psi))
            
        end

        flush(stdout)
        
    end
    
    for i in 1:2:N-1

        s1 = sites[i]
        s2 = sites[i+1]

        hg=haar_gate(s1,s2)
        orthogonalize!(psi,i)

        wf = (psi[i] * psi[i+1]) * hg
        noprime!(wf)

        inds_i = uniqueinds(psi[i],psi[i+1])
        U,S,V = svd(wf,inds_i,cutoff=1E-16)
        psi[i] = U
        psi[i+1] = S*V
    end
    
    
    
    
    
    println("Before entropy calculation ",norm(psi))
            
    orthogonalize!(psi, b)
    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
    println(S)
    S0=0
    SvN = 0.0
    for n=1: ITensors.dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log(2,p)
      if(S[n,n]>1E-16)
          S0=S0+1
      end
    end
    
  
    


    push!(data_S02,S0)
    push!(data_SvN2,SvN)
    
    println(f,S0," ",SvN)
    
    println("trial ",trial," S0 ",ITensors.dim(S, 1)," Cutoff S_0 ",S0," SvN ",SvN)
    
    
 end


println("///////////////////",mean(log2.(data_S02)))
end
            



