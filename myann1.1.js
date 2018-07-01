function Table(w,h){
	this.w=w;
	this.h=h;
	var i;
	this.wh=new Array(w);
	for(i=0;i<w;i++)
		this.wh[i]=new Array(h);
}

function Tensor(w,h,ww,hh){
	this.w=w;
	this.h=h;
	this.ww=ww;
	this.hh=hh;
	var i,j;
	this.xy=new Array(w);
	for(i=0;i<w;i++){
		this.xy[i]=new Array(h);
		for(j=0;j<h;j++)
			this.xy[i][j]=new Table(ww,hh);
	}
}

function Layer(dd,ww,hh){
	Tensor.call(this,2,dd,ww,hh);
	this.prop=this.xy[0];
	this.back=this.xy[1];
}

function Weight(w,h,ww,hh){
	Tensor.call(this,w,h,ww,hh);
	this.bias=new Array(h);
	var i,j,ii,jj;
	for(i=0;i<this.w;i++)
		for(j=0;j<this.h;j++)
			for(ii=0;ii<this.ww;ii++)
				for(jj=0;jj<this.hh;jj++)
					this.xy[i][j].wh[ii][jj]=Math.sqrt(24/(this.w+this.h))*(Math.random()-0.5);
	for(j=0;j<this.h;j++)
		this.bias[j]=Math.sqrt(24/(this.w+this.h))*(Math.random()-0.5);
}
/*
Tensor.prototype.layer=function(){
	this.prop=this.xy[0];
	this.back=this.xy[1];
}
Tensor.prototype.weight = function() {
	this.bias=new Array(h);
	var i,j,ii,jj;
	for(i=0;i<this.w;i++)
		for(j=0;j<this.h;j++)
			for(ii=0;ii<this.ww;ii++)
				for(jj=0;jj<this.hh;jj++)
					this.xy[i][j].wh[ii][jj]=Math.sqrt(24/(this.w*this.w+this.h*this.h))*(Math.random()-0.5);
	for(j=0;j<this.h;j++)
		this.bias[j]=Math.sqrt(24/(this.w*this.w+this.h*this.h))*(Math.random()-0.5);
};*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
var Ann=(function(){
	return{
		
		inisample:function(Inp,Out){
			var i,j,ii,jj,temp,num=14;
			for(i=0;i<Inp.w;i++){
				temp=i%num;
				for(j=0;j<Inp.h;j++)
					for(ii=0;ii<Inp.ww;ii++)
						for(jj=0;jj<Inp.hh;jj++){
							if (jj==0)
								Inp.xy[i][j].wh[ii][jj]=1;
							else if(Math.random()<14/26)
								Inp.xy[i][j].wh[ii][jj]=Math.floor(Math.random()*255);
							else 
								Inp.xy[i][j].wh[ii][jj]=temp*255/14;
						}		
				for(j=0;j<Out.h;j++)
					for(ii=0;ii<Out.ww;ii++)
						for(jj=0;jj<Out.hh;jj++)
							Out.xy[i][j].wh[ii][jj]=(j==temp);
			}
		},
		insertsample:function(Inp,num,c,a,b,value){
			Inp.xy[num][c].wh[a][b]=value;
		},
		/*
		turnprop:function(X,Inp,s,dv){
			var i,j,k;
			for(k=0;k<X.h;k++)
				for(i=0;i<X.ww;i++)
					for(j=0;j<X.hh;j++)
						X.prop[k].wh[i][j]=Inp.xy[s][k].wh[i][j]/dv;
		},
		*/
		turnprop:function(X,Inp,s,dv,nbool){
			var i,j,k;
			
			var noise=new Array(6),ni,nj;
			for(i=0;i<6;i++)
				if (nbool){
					if(i%3!=2){
						if (s%14<10)
							noise[i]=(Math.random()-0.5)*(1.05/21);
						else if(s%14==10)
							noise[i]=(Math.random()-0.5)*(16.8/21);
						else
							noise[i]=(Math.random()-0.5)*(4.2/21);
					}
					else
						noise[i]=X.ww*0.5+(Math.random()-0.5)*(4/3);
					}
				else{
					if(i%3!=2)
						noise[i]=0;
					else
						noise[i]=14;
					}
			for(k=0;k<X.h;k++){
				for(i=0;i<X.ww;i++)
					for(j=0;j<X.hh;j++)
						X.prop[k].wh[i][j]=0;

				for(i=0;i<X.ww;i++)
					for(j=0;j<X.hh;j++){
						ni=i-14;
						nj=j-14;
						ni=Math.floor((1+noise[0])*ni+noise[1]*nj+noise[2]);
						nj=Math.floor(noise[3]*ni+(1+noise[4])*nj+noise[5]);
						if (ni<0 || ni>=X.ww)continue;
						if (nj<0 || nj>=X.hh)continue;
						X.prop[k].wh[ni][nj]=Inp.xy[s][k].wh[i][j]/dv;
					}
			}
		},
		conv:function(X,XnY,Y,a,padd){//CNN propagation
			var i,j,k,ii,jj,kk,sum,offset=-Math.floor(XnY.ww*0.5)*padd,oi,oj;
			for(k=0;k<Y.h;k++)
				for(i=0;i<Y.ww;i++)
					for(j=0;j<Y.hh;j++){
						sum=XnY.bias[k];
						for(kk=0;kk<X.h;kk++){
							for(ii=0;ii<XnY.ww;ii++){
								oi=i+offset+ii;
								if (oi<0 || oi>=X.ww) continue;
								for(jj=0;jj<XnY.hh;jj++){
									oj=j+offset+jj;
									if (oj<0 || oj>=X.hh) continue;
									sum+=X.prop[kk].wh[oi][oj]*XnY.xy[kk][k].wh[ii][jj];
								}
							}
						}
						switch(a){
							case 0: case 1: Y.prop[k].wh[i][j]=sum;break;
							case 2:Y.prop[k].wh[i][j]=Math.max(sum,0);break;
						}
					}
		},
		pool:function(X,Y){
			var i,j,k,sum;
			for(k=0;k<Y.h;k++)
				for(i=0;i<Y.ww;i++)
					for(j=0;j<Y.hh;j++){
						sum=Math.max(
							X.prop[k].wh[i*2][j*2],
							X.prop[k].wh[i*2][j*2+1],
							X.prop[k].wh[i*2+1][j*2],
							X.prop[k].wh[i*2+1][j*2+1]
						);
						Y.prop[k].wh[i][j]=sum;
					}
		},
		//conc:function(Xarr,Y){},//concat/slice
		turnback:function(X,Out,s,ss){//softmax 레이어+backpropagation 전처리
			var i,j,k;
			var smmax=-99999,smsum=0,smp=[0,0,0];//소프트맥스 변수
			for(k=0;k<X.h;k++)
				for(i=0;i<X.ww;i++)
					for(j=0;j<X.hh;j++)
						if (smmax<X.prop[k].wh[i][j]){
							smmax=X.prop[k].wh[i][j];
							smp=[k,i,j];
						}
			for(k=0;k<X.h;k++)
				for(i=0;i<X.ww;i++)
					for(j=0;j<X.hh;j++){
						X.prop[k].wh[i][j]=Math.exp((X.prop[k].wh[i][j]-smmax)*ss);
						smsum+=X.prop[k].wh[i][j];
					}
			if (s!=-1)
				for(k=0;k<X.h;k++)
					for(i=0;i<X.ww;i++)
						for(j=0;j<X.hh;j++){
							X.prop[k].wh[i][j]/=smsum;
							X.back[k].wh[i][j]=X.prop[k].wh[i][j]-Out.xy[s][k].wh[i][j];
						}
			return smp;//최종 출력(가장높은 값)의 좌표
		},
		pool_b:function(X,Y){//pooling backpropagation
			var i,j,k,temp,temb;
			for(k=0;k<Y.h;k++)
				for(i=0;i<Y.ww;i++)
					for(j=0;j<Y.hh;j++){
						temp=Y.prop[k].wh[i][j];
						temb=Y.back[k].wh[i][j];
						X.back[k].wh[i*2][j*2]=(X.prop[k].wh[i*2][j*2]==temp)*temb;
						X.back[k].wh[i*2][j*2+1]=(X.prop[k].wh[i*2][j*2+1]==temp)*temb;
						X.back[k].wh[i*2+1][j*2]=(X.prop[k].wh[i*2+1][j*2]==temp)*temb;
						X.back[k].wh[i*2+1][j*2+1]=(X.prop[k].wh[i*2+1][j*2+1]==temp)*temb;
					}				
		},
		conv_b:function(X,XnY,Y,a,padd,lr){//CNN backpropagation
			var i,j,k,ii,jj,kk,sum,offset=-Math.floor(XnY.ww*0.5)*(2-padd),oi,oj;
			for(k=0;k<Y.h;k++){
				sum=0;
				for(i=0;i<Y.ww;i++)
					for(j=0;j<Y.hh;j++){
						switch(a){
							case 0: Y.back[k].wh[i][j]=0;break;
							case 2:if (Y.prop[k].wh[i][j]<=0) Y.back[k].wh[i][j]=0;break;
						}
						sum+=-lr*Y.back[k].wh[i][j];
					}
				XnY.bias[k]+=-lr*sum*XnY.ww*XnY.hh;//Y.back[k].wh[i][j]/=XnY.h;			
			}
			for(k=0;k<X.h;k++){
				for(i=0;i<X.ww;i++)
					for(j=0;j<X.hh;j++){
						sum=0;
						for(kk=0;kk<Y.h;kk++)
							for(ii=0;ii<XnY.ww;ii++){
								oi=i+offset+ii;
								if (oi<0 || oi>=Y.ww)continue;
								for(jj=0;jj<XnY.hh;jj++){
									oj=j+offset+jj;
									if (oj<0 || oj>=Y.hh)continue;
									XnY.xy[k][kk].wh[XnY.ww-1-ii][XnY.hh-1-jj]+=-lr*X.prop[k].wh[i][j]*Y.back[kk].wh[oi][oj];
									sum+=Y.back[kk].wh[oi][oj]*XnY.xy[k][kk].wh[XnY.ww-1-ii][XnY.hh-1-jj];
								}
							}
						X.back[k].wh[i][j]=sum;
					}
			}
		},
		get_pred:function(ans,Out,s){//정답여부 확인
			return Out.xy[s][ans[0]].wh[ans[1]][ans[2]];
		},
		get_rms:function(X){//출력값에대한 RMS오차(평균제곱근)
			var k,i,j,sum=0.0;
			for(k=0;k<X.h;k++)
				for(i=0;i<X.ww;i++)
					for(j=0;j<X.hh;j++)
					sum+=X.back[k].wh[i][j]*X.back[k].wh[i][j];
			sum/=X.hh*X.w*X.h;
			return Math.sqrt(sum);
		},
		get_Xval:function(X,type){
			var k,arr=new Array(X.h);
			for(k=0;k<X.h;k++)
				arr[k]=X.xy[type][k].wh[0][0].toFixed(5);
			return arr;
		}
	};
})();
///////////////////////////////////////////////////////////////////////////////////
console.log('ann init');
