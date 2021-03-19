# ML-Robustness-Tutorial-Hitachi-ABB
ML Robustness Tutorial at Hitachi ABB Power Grids

This exercise is based on the first part of the tutorial, the goal is to apply the understanding of the Box and MILP based analysis for verifying the robustness of networks with Maxpool activations. 

Remember the most precise and sound box transformer for y = max(0,x) where l<=x<=u is [max(0,l), max(0,u)].

The corresponding MILP transformer adds the following constraints:

y<= x- l.(1-a),

y>=x,

y<=u.a,

y>=0,

a \in {0,1}.
