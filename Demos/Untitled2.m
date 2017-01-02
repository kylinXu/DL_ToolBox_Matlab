pro=[0.1,0.3,0.5,0.1];

val=[3,5,9,7];
aaa=zeros(100,4);
for i=1:100

aaa(i,:)=mnrnd(1,pro);

end

[ia,ib]=find(aaa(:,3)==1);
[ia,ib]=find(aaa(:,1)==1);
[ia,ib]=find(aaa(:,2)==1);
[ia,ib]=find(aaa(:,4)==1);



find(aaa(:)==1)

bbb=mnrnd(1,pro);

ccc=find(bbb==1);