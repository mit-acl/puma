function result=getSimpsonCoeff(j,total)

    j_is_even = (rem(j, 2) == 0);

    %See Composite Simpson's rule in https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule

    if(j==1 || j==total)  %Beginning or end
        result=1.0;
    elseif (j_is_even) %Note that Matlab is 1-indexing (that's why this changes wrt to the above link)
        result=4.0;
    else 
        result=2.0;
    end

end
