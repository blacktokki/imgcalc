var Mycv=(function(){//메소드 모듈
	var ranger=new Array(6);//x1,y1,x2,y2,w/2,h/2
	var searcher=function(arr,i,j){//BFS area search
		if(!(i<0 || j<0 || i>=arr.length||j>=arr[0].length))
			if (arr[i][j]!=0){
				arr[i][j]=0;
				ranger[0]=Math.min(ranger[0],i);
				ranger[1]=Math.min(ranger[1],j);
				ranger[2]=Math.max(ranger[2],i);
				ranger[3]=Math.max(ranger[3],j);
				searcher(arr,i-1,j-1);
				searcher(arr,i-1,j);
				searcher(arr,i-1,j+1);
				searcher(arr,i,j-1);
				searcher(arr,i,j+1);
				searcher(arr,i+1,j-1);
				searcher(arr,i+1,j);
				searcher(arr,i+1,j+1);
				}
	}
	return{
		presearch:function(data,arr,width){//preprocess image_data를 arr로 복사 1d->2d
			var i=-1,j=width-1,k,swi=0,numj=0,temp;
			var obj=new Object();
			obj.lowi=new Array();
			obj.higi=new Array();
			for(k=0;k<data.length;k+=4){//문자열 개수 감지
				j++;
				if (j==width){
					j=0;
					if (numj>0){
						if (swi==0){
							swi=1;
							obj.lowi.push(i);
						}
						numj=0;
					}
					else if(swi==1){
						swi=0;
						obj.higi.push(i);
					}
					i++;
					arr[i]=new Array(width);
				}
				temp=(data[k]+data[k+1]+data[k+2])/3;
				temp-=temp%128;//grayscale+이진화
				data[k+1] =temp;
				data[k+2] =temp;
				data[k+3] = 255;
				arr[i][j]=temp;
				if (temp!=0)
					numj++;
			}
			if(swi==1)
				obj.higi.push(i);
			obj.size=obj.higi.length;
			return obj;
		},	
		areasearch:function(arr,i,j){//영역 감지기
			ranger[0]=i;
			ranger[1]=j;
			ranger[2]=i;
			ranger[3]=j;
			searcher(arr,i,j);
			ranger[4]=(ranger[3]-ranger[1])/2;
			ranger[5]=(ranger[2]-ranger[0])/2;
			return ranger;
		}
	};
})();
console.log('mycv init');