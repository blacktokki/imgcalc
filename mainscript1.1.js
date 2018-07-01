var num_sp=14*1024;
var rp=0,sum_rms=0,sum_pred=0,sp,spm,sco_i,sco_s=0,sco_a,Gval;
//lenet-5
var I=new Tensor(num_sp,1,28,28);
var Itest=new Tensor(64,1,28,28);
var A=new Layer(1,28,28);
var AnB=new Weight(1,6,5,5);
var B=new Layer(6,28,28);
var C=new Layer(6,14,14);
var CnD=new Weight(6,16,5,5);
var D=new Layer(16,10,10);
var E=new Layer(16,5,5);
var EnF=new Weight(16,120,5,5);
var F=new Layer(120,1,1);
var FnG=new Weight(120,84,1,1);
var G=new Layer(84,1,1);
var GnH=new Weight(84,14,1,1);
var H=new Layer(14,1,1);
var O=new Tensor(num_sp,14,1,1);
var Sco=new Array(),Sco2;
for(sco_i=0;sco_i<16;sco_i++)
	Sco[sco_i]=Array.apply(null, Array(14)).map(Number.prototype.valueOf,0);
Sco[16]=['0','1','2','3','4','5','6','7','8','9','+','-','*','/'];
Ann.inisample(I,O);

function train(){
	sp=rp%num_sp;//Math.floor(Math.random()*num_sp);
	spm=sp%14;
	
	Ann.turnprop(A,I,sp,255,1);
	Ann.conv(A,AnB,B,2,1);
	Ann.pool(B,C);
	Ann.conv(C,CnD,D,2,0);
	Ann.pool(D,E);
	Ann.conv(E,EnF,F,2,0);
	Ann.conv(F,FnG,G,2,1);
	Ann.conv(G,GnH,H,1,1);
	Gval=Ann.turnback(H,O,sp,1);
	Sco[Gval[0]][spm]++;
	Sco[14][spm]++;
	Sco[15][spm]=Sco[spm][spm]/Sco[14][spm];
	sco_s+=Sco[15][spm];
	sco_a=sco_s/(sp+1);
	Sco[15][spm]=Sco[15][spm].toFixed(3);
	sum_pred+=Ann.get_pred(Gval,O,sp);
	sum_rms+=Ann.get_rms(H);
	if(rp%351==0){
		//console.log(A.prop);
		console.log(new Date().toLocaleTimeString(),":",rp);
		console.log("Acc:",JSON.stringify(Sco[15]));
		console.log("Pred:",JSON.stringify(Ann.get_Xval(H,0)));
		console.log("Grad:",JSON.stringify(Ann.get_Xval(H,1)));
		($('#canvas3').get(0)).getContext('2d').fillRect(0,0,28,28);
		var i,j;
		for(i=0;i<A.ww;i++)
			for(j=0;j<A.hh;j++)
				if(A.prop[0].wh[i][j]!=0)
				($('#canvas3').get(0)).getContext('2d').strokeRect(i,j,1,1);
	}
	if (sp==num_sp-1){
		console.log(":"+new Date().toLocaleTimeString()+":"+(sum_pred/num_sp).toFixed(6)+":"+(sum_rms/num_sp).toFixed(6)+":"+rp);
		sum_rms=0;
		sum_pred=0;
		sco_s=0;
		for(sco_i=0;sco_i<16;sco_i++){
			console.log(JSON.stringify(Sco[sco_i]));
			Sco[sco_i]=Array.apply(null, Array(14)).map(Number.prototype.valueOf,0);
		}
	}
	if (Sco[spm][spm]/Sco[14][spm]<0.95){
		Ann.conv_b(G,GnH,H,1,1,0.000001);
		Ann.conv_b(F,FnG,G,2,1,0.000001);
		Ann.conv_b(E,EnF,F,2,0,0.000002);
		Ann.pool_b(D,E)
		Ann.conv_b(C,CnD,D,2,0,0.0001);
		Ann.pool_b(B,C);
		Ann.conv_b(A,AnB,B,2,1,0.0001);
	}
	rp=(rp+1);//++;
}
function test(srp){
	Ann.turnprop(A,Itest,srp,255,0);
	Ann.conv(A,AnB,B,2,1);
	Ann.pool(B,C);
	Ann.conv(C,CnD,D,2,0);
	Ann.pool(D,E);
	Ann.conv(E,EnF,F,2,0);
	Ann.conv(F,FnG,G,2,1);
	Ann.conv(G,GnH,H,1,1);
	Gval=Ann.turnback(H,O,-1,1);
	return Gval[0];
}
/////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
$(function(){
	$.ajax({
		url: "sample.json",
		beforeSend: function(xhr){
			if(xhr.overrideMimeType)
				xhr.overrideMimeType("application/json");
		},
		dataType: 'json',
		success: function(data){
			I=data.I;
			O=data.O;
			console.log("sample.json init");
			data=null;
		}
	});
	$.ajax({
		url: "weight.json",
		beforeSend: function(xhr){
			if(xhr.overrideMimeType)
				xhr.overrideMimeType("application/json");
		},
		dataType: 'json',
		success: function(data2){
			AnB=data2.AnB;
			CnD=data2.CnD;
			EnF=data2.EnF;
			FnG=data2.FnG;
			GnH=data2.GnH;
			console.log("weight.json init");
			data2=null;
		}
	});
	var canvas1=canvas1=$('#canvas1').get(0),ctx1,ctx2,ctx3,ctx4
	var m_width=canvas1.width,m_height=canvas1.height;
	var ctx_init=function(){
		ctx1=canvas1.getContext('2d');
		ctx2=($('#canvas2').get(0)).getContext('2d');
		ctx3=($('#canvas3').get(0)).getContext('2d');
		ctx4=($('#canvas4').get(0)).getContext('2d');
		ctx1.strokeStyle = $('#colors').get(0).value;
		ctx1.lineWidth = "5";
		ctx1.lineCap = "round";
		ctx2.strokeStyle = "red";
		ctx2.fillStyle = "white";
		ctx3.strokeStyle = "white";
	};
	ctx_init();
	var client=1;
	if (client){
		$('#viewer').hide();
		$('#viewButton').hide();
	}
	else
		var main=setInterval(train,3);////////////////////////////////////////////////trainer
	$("#canvas1").on('mousedown',function(e){
		ctx1.beginPath();
		ctx1.moveTo(e.pageX -canvas1.offsetLeft, e.pageY-canvas1.offsetTop);
		$("#canvas1").on('mousemove',function(e){
			ctx1.lineTo(e.pageX -canvas1.offsetLeft, e.pageY-canvas1.offsetTop);
			ctx1.stroke();
		});
		$("#canvas1").on('mouseup',function(){
			$(this).off('mousemove');
			if ($("#auto").get(0).checked)
				$('#runButton').click();
		});
	});

	//button event
	$('#reset').on('click', function () {
		ctx1.fillRect(0,0,m_width*2,m_height*2);
		ctx2.fillRect(0,0,m_width,m_height);
		ctx3.fillRect(0,0,28,28);
		ctx4.fillRect(0,0,m_width,28);
		$("#str").text('수식:');
		$("#strr").text('결과:');
	});
	$("#reset").click();
	$('#inp_w').on('change', function(e){
		m_width=e.target.value;
		canvas1.width=m_width;
		$('#canvas2').get(0).width=m_width;
		$('#canvas4').get(0).width=m_width;
		ctx_init();
		$("#reset").click();
		$('#viewer').css("width",m_width);
		
	});
	$('#inp_h').on('change', function(e){
		m_height=e.target.value;
		canvas1.height=m_height;
		$('#canvas2').get(0).height=m_height;
		ctx_init();
		$("#reset").click();
		$('#viewer').height=m_height;
		m_height=m_height;
	});
	$('#colors').on('change', function(e){
		ctx1.strokeStyle = e.target.value;
	});
	$('#loadButton').on('change', function(e){
		var file = e.target.files[0];
		var fileReader = new FileReader();
		fileReader.onload = function(e){
			var image = new Image();
			image.src = e.target.result;
			image.onload = function(){
				image.width=m_width;
				image.height=m_height;
				ctx1.drawImage(image, 0, 0, image.width, image.height);
			}
		};
		fileReader.readAsDataURL(file);
	});

	$('#runButton').on('click', function () {  
		var d = ctx1.getImageData(0,0, m_width,m_height).data;
		var arr=new Array(256);
		var pre=Mycv.presearch(d,arr,m_width);
		var str="",c=0,temp,i,j,k,mni,mxi;
		$("#str").text('수식:');
		$("#strr").text('결과:');
		for(k=0;k<pre.size;k++){
			mni=pre.lowi[k];
			mxi=pre.higi[k];
			for(j=0;j<m_width;j++)
			for(i=mni;i<mxi;i++)
			if(arr[i][j]==128){
				var a2=Mycv.areasearch(arr,i,j);
				if(a2[5]/a2[4]<0.25){
					a2[0]-=a2[4]-a2[5];
					a2[2]+=a2[4]-a2[5];
					a2[5]=(a2[2]-a2[0])/2;
					j=a2[3];
				}
				temp=Math.max(a2[4],a2[5]);
				var px=a2[1]+a2[4]-temp;
				var py=a2[0]+a2[5]-temp;
				var dd,kk,ii=-1,jj=27;
				temp*=2;
				ctx2.drawImage(canvas1,px,py,temp,temp,px,py,temp,temp);
				ctx2.strokeRect(px,py,temp, temp);
				ctx3.fillRect(0,0,28,28);
				ctx3.drawImage(canvas1,px,py,temp,temp,3.5,3.5,21,21);
				dd=ctx3.getImageData(0,0,28,28).data;
				ctx4.fillRect(c*28,0,28,28);
				ctx4.drawImage(canvas1,px,py,temp,temp,3.5+28*c,3.5,21,21);
				for(kk=0; kk<dd.length;kk+=4){
					jj++;
					if (jj==28){
						jj=0;
						ii++;
					}
					temp=dd[kk]>127;
					Ann.insertsample(Itest,c,0,jj,ii,temp*255);
				}
				/*for (var v=0;v<28;v++)
					console.log(":"+JSON.stringify(Itest.front[c][0].back[v]));*/
				temp=test(c++);
				str=str+Sco[16][temp];
			}
		}
		$("#str").text("수식:"+str);
		try{
			$("#strr").text("결과:"+eval(str));
		}
		catch(exception){
			$("#strr").text("결과:"+exception);
		}
	});
	$('#viewButton').on('click', function () {
		$('#viewer').toggle('slow');
	});
	$('#weightButton').on('click', function () {
		var obj=new Object();
		obj.AnB=AnB;
		obj.CnD=CnD;
		obj.EnF=EnF;
		obj.FnG=FnG;//++
		obj.GnH=GnH;
		var obj_data="data:application/json;charset=utf-8,"+ encodeURIComponent(JSON.stringify(obj));//,null,\t
		$("#links").attr("href",obj_data);
		$("#links").attr("download","weight.json");
		$("#links").css("display","");
	});
});		