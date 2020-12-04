Information gain is calculated for each attribute and splitting attribute is decided.
Recursive calls are made at each level.

Synthetic Dataset:
30 data points in total. 
Attribute 1 - { 1,2,3}
Attribute 2 - { 1,2,3}
Attribute 3 - { 1,2,3,4,5}
2 class values are present - {0/1 } 

Output:

Simple:

Number of Data Elements 30
Number of attributes 5
Number of class values 2
Attribute 1 Attribute Value 1 Data Count 10.000000
Attribute 1 Attribute Value 2 Data Count 10.000000
Attribute 1 Attribute Value 3 Data Count 10.000000
Gain: 0.189340, Attr: 1
Attribute 2 Attribute Value 1 Data Count 10.000000
Attribute 2 Attribute Value 2 Data Count 10.000000
Attribute 2 Attribute Value 3 Data Count 10.000000
Gain: 0.031705, Attr: 2
Attribute 3 Attribute Value 1 Data Count 6.000000
Attribute 3 Attribute Value 2 Data Count 6.000000
Attribute 3 Attribute Value 3 Data Count 6.000000
Attribute 3 Attribute Value 4 Data Count 6.000000
Attribute 3 Attribute Value 5 Data Count 6.000000
Gain: 0.469322, Attr: 3
level 0 Splitting Attribute:3
Recursive call -> level 1: attribute 3 attributeVal 1 #data Points 6:
Attribute 1 Attribute Value 1 Data Count 2.000000
Attribute 1 Attribute Value 2 Data Count 2.000000
Attribute 1 Attribute Value 3 Data Count 2.000000
Gain: 0.918296, Attr: 1
Attribute 2 Attribute Value 1 Data Count 6.000000
Gain: 0.000000, Attr: 2
Gain: 0.000000, Attr: 3
level 1 Splitting Attribute:1
Recursive call -> level 2: attribute 1 attributeVal 1 #data Points 2:
Leaf -> level 2, #data Points 2 Class Label 1:
Recursive call -> level 2: attribute 1 attributeVal 2 #data Points 2:
Leaf -> level 2, #data Points 2 Class Label 0:
Recursive call -> level 2: attribute 1 attributeVal 3 #data Points 2:
Leaf -> level 2, #data Points 2 Class Label 1:
Recursive call -> level 1: attribute 3 attributeVal 2 #data Points 6:
Recursive call -> level 1: attribute 3 attributeVal 3 #data Points 6:
Recursive call -> level 1: attribute 3 attributeVal 4 #data Points 6:
Attribute 1 Attribute Value 1 Data Count 2.000000
Attribute 1 Attribute Value 2 Data Count 2.000000
Attribute 1 Attribute Value 3 Data Count 2.000000
Gain: 0.918296, Attr: 1
Attribute 2 Attribute Value 2 Data Count 2.000000
Attribute 2 Attribute Value 3 Data Count 4.000000
Gain: 0.251629, Attr: 2
Gain: 0.251629, Attr: 3
level 1 Splitting Attribute:1
Recursive call -> level 2: attribute 1 attributeVal 1 #data Points 2:
Leaf -> level 2, #data Points 2 Class Label 1:
Recursive call -> level 2: attribute 1 attributeVal 2 #data Points 2:
Leaf -> level 2, #data Points 2 Class Label 0:
Recursive call -> level 2: attribute 1 attributeVal 3 #data Points 2:
Leaf -> level 2, #data Points 2 Class Label 1:
Recursive call -> level 1: attribute 3 attributeVal 5 #data Points 6:
time:0.001102
in testing
Correct 30 Incorrect 0 Unexpected 0

Parallel:

rank 0 in output time computation 0.000159
rank 1 in output time computation 0.000162
rank 2 in output time computation 0.000162
rank 3 in output time computation 0.000162
Correct 30 Incorrect 0 Unexpected 0
Correct 30 Incorrect 0 Unexpected 0
Correct 30 Incorrect 0 Unexpected 0
Correct 30 Incorrect 0 Unexpected 0
