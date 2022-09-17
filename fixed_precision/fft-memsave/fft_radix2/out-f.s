/home/mazzoni/Pulp/pulp-sdk/pkg/pulp_riscv_gcc/1.0.14/bin/riscv32-unknown-elf-objdump -Mmarch=rv32imcxgap9 /home/mazzoni/Pulp/fft/fft_radix2/build/vega/test/test -d

/home/mazzoni/Pulp/fft/fft_radix2/build/vega/test/test:     file format elf32-littleriscv


Disassembly of section .vectors:

1c008000 <__irq_vector_base>:
1c008000:	5790006f          	j	1c008d78 <__rt_illegal_instr>
1c008004:	0900006f          	j	1c008094 <__rt_no_irq_handler>
1c008008:	08c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00800c:	0880006f          	j	1c008094 <__rt_no_irq_handler>
1c008010:	0840006f          	j	1c008094 <__rt_no_irq_handler>
1c008014:	0800006f          	j	1c008094 <__rt_no_irq_handler>
1c008018:	07c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00801c:	0780006f          	j	1c008094 <__rt_no_irq_handler>
1c008020:	0740006f          	j	1c008094 <__rt_no_irq_handler>
1c008024:	0700006f          	j	1c008094 <__rt_no_irq_handler>
1c008028:	06c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00802c:	0680006f          	j	1c008094 <__rt_no_irq_handler>
1c008030:	0640006f          	j	1c008094 <__rt_no_irq_handler>
1c008034:	0600006f          	j	1c008094 <__rt_no_irq_handler>
1c008038:	05c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00803c:	0580006f          	j	1c008094 <__rt_no_irq_handler>
1c008040:	0540006f          	j	1c008094 <__rt_no_irq_handler>
1c008044:	0500006f          	j	1c008094 <__rt_no_irq_handler>
1c008048:	04c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00804c:	0480006f          	j	1c008094 <__rt_no_irq_handler>
1c008050:	0440006f          	j	1c008094 <__rt_no_irq_handler>
1c008054:	0400006f          	j	1c008094 <__rt_no_irq_handler>
1c008058:	03c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00805c:	0380006f          	j	1c008094 <__rt_no_irq_handler>
1c008060:	0340006f          	j	1c008094 <__rt_no_irq_handler>
1c008064:	0300006f          	j	1c008094 <__rt_no_irq_handler>
1c008068:	02c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00806c:	0280006f          	j	1c008094 <__rt_no_irq_handler>
1c008070:	0240006f          	j	1c008094 <__rt_no_irq_handler>
1c008074:	0200006f          	j	1c008094 <__rt_no_irq_handler>
1c008078:	01c0006f          	j	1c008094 <__rt_no_irq_handler>
1c00807c:	0180006f          	j	1c008094 <__rt_no_irq_handler>

1c008080 <_start>:
1c008080:	3490006f          	j	1c008bc8 <_entry>
1c008084:	4f50006f          	j	1c008d78 <__rt_illegal_instr>
	...

1c008090 <__rt_debug_struct_ptr>:
1c008090:	1584                	addi	s1,sp,736
1c008092:	1c00                	addi	s0,sp,560

1c008094 <__rt_no_irq_handler>:
1c008094:	0000006f          	j	1c008094 <__rt_no_irq_handler>

1c008098 <__rt_semihosting_call>:
1c008098:	00100073          	ebreak
1c00809c:	00008067          	ret

Disassembly of section .text:

1c0080a0 <cosf>:
1c0080a0:	1101                	addi	sp,sp,-32
1c0080a2:	3f491737          	lui	a4,0x3f491
1c0080a6:	ce06                	sw	ra,28(sp)
1c0080a8:	c1f536b3          	p.bclr	a3,a0,0,31
1c0080ac:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x23479e38>
1c0080b0:	00000593          	li	a1,0
1c0080b4:	02d75663          	ble	a3,a4,1c0080e0 <cosf+0x40>
1c0080b8:	7f800737          	lui	a4,0x7f800
1c0080bc:	00e6c763          	blt	a3,a4,1c0080ca <cosf+0x2a>
1c0080c0:	08a57553          	fsub.s	a0,a0,a0
1c0080c4:	40f2                	lw	ra,28(sp)
1c0080c6:	6105                	addi	sp,sp,32
1c0080c8:	8082                	ret
1c0080ca:	002c                	addi	a1,sp,8
1c0080cc:	2841                	jal	1c00815c <__ieee754_rem_pio2f>
1c0080ce:	fa2537b3          	p.bclr	a5,a0,29,2
1c0080d2:	45b2                	lw	a1,12(sp)
1c0080d4:	4522                	lw	a0,8(sp)
1c0080d6:	0017a763          	p.beqimm	a5,1,1c0080e4 <cosf+0x44>
1c0080da:	0027ab63          	p.beqimm	a5,2,1c0080f0 <cosf+0x50>
1c0080de:	eb99                	bnez	a5,1c0080f4 <cosf+0x54>
1c0080e0:	2ce1                	jal	1c0083b8 <__kernel_cosf>
1c0080e2:	b7cd                	j	1c0080c4 <cosf+0x24>
1c0080e4:	4605                	li	a2,1
1c0080e6:	101000ef          	jal	ra,1c0089e6 <__kernel_sinf>
1c0080ea:	20a51553          	fneg.s	a0,a0a0
1c0080ee:	bfd9                	j	1c0080c4 <cosf+0x24>
1c0080f0:	24e1                	jal	1c0083b8 <__kernel_cosf>
1c0080f2:	bfe5                	j	1c0080ea <cosf+0x4a>
1c0080f4:	4605                	li	a2,1
1c0080f6:	0f1000ef          	jal	ra,1c0089e6 <__kernel_sinf>
1c0080fa:	b7e9                	j	1c0080c4 <cosf+0x24>

1c0080fc <sinf>:
1c0080fc:	1101                	addi	sp,sp,-32
1c0080fe:	3f491737          	lui	a4,0x3f491
1c008102:	ce06                	sw	ra,28(sp)
1c008104:	c1f536b3          	p.bclr	a3,a0,0,31
1c008108:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x23479e38>
1c00810c:	00d74863          	blt	a4,a3,1c00811c <sinf+0x20>
1c008110:	00000593          	li	a1,0
1c008114:	4601                	li	a2,0
1c008116:	0d1000ef          	jal	ra,1c0089e6 <__kernel_sinf>
1c00811a:	a039                	j	1c008128 <sinf+0x2c>
1c00811c:	7f800737          	lui	a4,0x7f800
1c008120:	00e6c763          	blt	a3,a4,1c00812e <sinf+0x32>
1c008124:	08a57553          	fsub.s	a0,a0,a0
1c008128:	40f2                	lw	ra,28(sp)
1c00812a:	6105                	addi	sp,sp,32
1c00812c:	8082                	ret
1c00812e:	002c                	addi	a1,sp,8
1c008130:	2035                	jal	1c00815c <__ieee754_rem_pio2f>
1c008132:	fa2537b3          	p.bclr	a5,a0,29,2
1c008136:	45b2                	lw	a1,12(sp)
1c008138:	4522                	lw	a0,8(sp)
1c00813a:	0017a763          	p.beqimm	a5,1,1c008148 <sinf+0x4c>
1c00813e:	0027a763          	p.beqimm	a5,2,1c00814c <sinf+0x50>
1c008142:	eb99                	bnez	a5,1c008158 <sinf+0x5c>
1c008144:	4605                	li	a2,1
1c008146:	bfc1                	j	1c008116 <sinf+0x1a>
1c008148:	2c85                	jal	1c0083b8 <__kernel_cosf>
1c00814a:	bff9                	j	1c008128 <sinf+0x2c>
1c00814c:	4605                	li	a2,1
1c00814e:	099000ef          	jal	ra,1c0089e6 <__kernel_sinf>
1c008152:	20a51553          	fneg.s	a0,a0a0
1c008156:	bfc9                	j	1c008128 <sinf+0x2c>
1c008158:	2485                	jal	1c0083b8 <__kernel_cosf>
1c00815a:	bfe5                	j	1c008152 <sinf+0x56>

1c00815c <__ieee754_rem_pio2f>:
1c00815c:	1101                	addi	sp,sp,-32
1c00815e:	3f491737          	lui	a4,0x3f491
1c008162:	cc22                	sw	s0,24(sp)
1c008164:	ce06                	sw	ra,28(sp)
1c008166:	ca26                	sw	s1,20(sp)
1c008168:	c84a                	sw	s2,16(sp)
1c00816a:	c1f53433          	p.bclr	s0,a0,0,31
1c00816e:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x23479e38>
1c008172:	00874763          	blt	a4,s0,1c008180 <__ieee754_rem_pio2f+0x24>
1c008176:	c188                	sw	a0,0(a1)
1c008178:	0005a223          	sw	zero,4(a1)
1c00817c:	4501                	li	a0,0
1c00817e:	a0a9                	j	1c0081c8 <__ieee754_rem_pio2f+0x6c>
1c008180:	4016d737          	lui	a4,0x4016d
1c008184:	be370713          	addi	a4,a4,-1053 # 4016cbe3 <__l2_shared_end+0x24155a43>
1c008188:	892a                	mv	s2,a0
1c00818a:	0a874163          	blt	a4,s0,1c00822c <__ieee754_rem_pio2f+0xd0>
1c00818e:	1c001737          	lui	a4,0x1c001
1c008192:	c6043433          	p.bclr	s0,s0,3,0
1c008196:	bdc72703          	lw	a4,-1060(a4) # 1c000bdc <PIo2+0x280>
1c00819a:	04a05863          	blez	a0,1c0081ea <__ieee754_rem_pio2f+0x8e>
1c00819e:	08e577d3          	fsub.s	a5,a0,a4
1c0081a2:	3fc91737          	lui	a4,0x3fc91
1c0081a6:	fd070713          	addi	a4,a4,-48 # 3fc90fd0 <__l2_shared_end+0x23c79e30>
1c0081aa:	02e40563          	beq	s0,a4,1c0081d4 <__ieee754_rem_pio2f+0x78>
1c0081ae:	1c001737          	lui	a4,0x1c001
1c0081b2:	be072703          	lw	a4,-1056(a4) # 1c000be0 <PIo2+0x284>
1c0081b6:	08e7f6d3          	fsub.s	a3,a5,a4
1c0081ba:	4505                	li	a0,1
1c0081bc:	08d7f7d3          	fsub.s	a5,a5,a3
1c0081c0:	c194                	sw	a3,0(a1)
1c0081c2:	08e7f7d3          	fsub.s	a5,a5,a4
1c0081c6:	c1dc                	sw	a5,4(a1)
1c0081c8:	40f2                	lw	ra,28(sp)
1c0081ca:	4462                	lw	s0,24(sp)
1c0081cc:	44d2                	lw	s1,20(sp)
1c0081ce:	4942                	lw	s2,16(sp)
1c0081d0:	6105                	addi	sp,sp,32
1c0081d2:	8082                	ret
1c0081d4:	1c001737          	lui	a4,0x1c001
1c0081d8:	be472703          	lw	a4,-1052(a4) # 1c000be4 <PIo2+0x288>
1c0081dc:	08e7f7d3          	fsub.s	a5,a5,a4
1c0081e0:	1c001737          	lui	a4,0x1c001
1c0081e4:	be872703          	lw	a4,-1048(a4) # 1c000be8 <PIo2+0x28c>
1c0081e8:	b7f9                	j	1c0081b6 <__ieee754_rem_pio2f+0x5a>
1c0081ea:	00e577d3          	fadd.s	a5,a0,a4
1c0081ee:	3fc91737          	lui	a4,0x3fc91
1c0081f2:	fd070713          	addi	a4,a4,-48 # 3fc90fd0 <__l2_shared_end+0x23c79e30>
1c0081f6:	02e40063          	beq	s0,a4,1c008216 <__ieee754_rem_pio2f+0xba>
1c0081fa:	1c001737          	lui	a4,0x1c001
1c0081fe:	be072703          	lw	a4,-1056(a4) # 1c000be0 <PIo2+0x284>
1c008202:	00e7f6d3          	fadd.s	a3,a5,a4
1c008206:	557d                	li	a0,-1
1c008208:	08d7f7d3          	fsub.s	a5,a5,a3
1c00820c:	c194                	sw	a3,0(a1)
1c00820e:	00e7f7d3          	fadd.s	a5,a5,a4
1c008212:	c1dc                	sw	a5,4(a1)
1c008214:	bf55                	j	1c0081c8 <__ieee754_rem_pio2f+0x6c>
1c008216:	1c001737          	lui	a4,0x1c001
1c00821a:	be472703          	lw	a4,-1052(a4) # 1c000be4 <PIo2+0x288>
1c00821e:	00e7f7d3          	fadd.s	a5,a5,a4
1c008222:	1c001737          	lui	a4,0x1c001
1c008226:	be872703          	lw	a4,-1048(a4) # 1c000be8 <PIo2+0x28c>
1c00822a:	bfe1                	j	1c008202 <__ieee754_rem_pio2f+0xa6>
1c00822c:	43491737          	lui	a4,0x43491
1c008230:	f8070713          	addi	a4,a4,-128 # 43490f80 <__l2_shared_end+0x27479de0>
1c008234:	84ae                	mv	s1,a1
1c008236:	0e874f63          	blt	a4,s0,1c008334 <__ieee754_rem_pio2f+0x1d8>
1c00823a:	039000ef          	jal	ra,1c008a72 <fabsf>
1c00823e:	1c0016b7          	lui	a3,0x1c001
1c008242:	1c001737          	lui	a4,0x1c001
1c008246:	87aa                	mv	a5,a0
1c008248:	bec72703          	lw	a4,-1044(a4) # 1c000bec <PIo2+0x290>
1c00824c:	bf06a503          	lw	a0,-1040(a3) # 1c000bf0 <PIo2+0x294>
1c008250:	46fd                	li	a3,31
1c008252:	50e7f743          	fmadd.s	a4,a5,a4,a0
1c008256:	c0071553          	fcvt.w.s	a0,a4,rtz
1c00825a:	1c001737          	lui	a4,0x1c001
1c00825e:	bdc72703          	lw	a4,-1060(a4) # 1c000bdc <PIo2+0x280>
1c008262:	d0057653          	fcvt.s.w	a2,a0
1c008266:	78e677cb          	fnmsub.s	a5,a2,a4,a5
1c00826a:	1c001737          	lui	a4,0x1c001
1c00826e:	be072703          	lw	a4,-1056(a4) # 1c000be0 <PIo2+0x284>
1c008272:	10e67753          	fmul.s	a4,a2,a4
1c008276:	08e7f5d3          	fsub.s	a1,a5,a4
1c00827a:	04a6c263          	blt	a3,a0,1c0082be <__ieee754_rem_pio2f+0x162>
1c00827e:	fff50693          	addi	a3,a0,-1
1c008282:	00269893          	slli	a7,a3,0x2
1c008286:	1c0006b7          	lui	a3,0x1c000
1c00828a:	5b868693          	addi	a3,a3,1464 # 1c0005b8 <npio2_hw>
1c00828e:	2116f683          	p.lw	a3,a7(a3)
1c008292:	ce043833          	p.bclr	a6,s0,7,0
1c008296:	02d80463          	beq	a6,a3,1c0082be <__ieee754_rem_pio2f+0x162>
1c00829a:	c08c                	sw	a1,0(s1)
1c00829c:	4094                	lw	a3,0(s1)
1c00829e:	08d7f7d3          	fsub.s	a5,a5,a3
1c0082a2:	08e7f7d3          	fsub.s	a5,a5,a4
1c0082a6:	c0dc                	sw	a5,4(s1)
1c0082a8:	f20950e3          	bgez	s2,1c0081c8 <__ieee754_rem_pio2f+0x6c>
1c0082ac:	20d696d3          	fneg.s	a3,a3a3
1c0082b0:	c094                	sw	a3,0(s1)
1c0082b2:	20f797d3          	fneg.s	a5,a5a5
1c0082b6:	40a00533          	neg	a0,a0
1c0082ba:	c0dc                	sw	a5,4(s1)
1c0082bc:	b731                	j	1c0081c8 <__ieee754_rem_pio2f+0x6c>
1c0082be:	0175d693          	srli	a3,a1,0x17
1c0082c2:	845d                	srai	s0,s0,0x17
1c0082c4:	ee86b6b3          	p.bclr	a3,a3,23,8
1c0082c8:	40d406b3          	sub	a3,s0,a3
1c0082cc:	4821                	li	a6,8
1c0082ce:	fcd856e3          	ble	a3,a6,1c00829a <__ieee754_rem_pio2f+0x13e>
1c0082d2:	1c001737          	lui	a4,0x1c001
1c0082d6:	be472703          	lw	a4,-1052(a4) # 1c000be4 <PIo2+0x288>
1c0082da:	78e676cb          	fnmsub.s	a3,a2,a4,a5
1c0082de:	08d7f7d3          	fsub.s	a5,a5,a3
1c0082e2:	78e6774b          	fnmsub.s	a4,a2,a4,a5
1c0082e6:	1c0017b7          	lui	a5,0x1c001
1c0082ea:	be87a783          	lw	a5,-1048(a5) # 1c000be8 <PIo2+0x28c>
1c0082ee:	70f67747          	fmsub.s	a4,a2,a5,a4
1c0082f2:	08e6f5d3          	fsub.s	a1,a3,a4
1c0082f6:	0175d793          	srli	a5,a1,0x17
1c0082fa:	ee87b7b3          	p.bclr	a5,a5,23,8
1c0082fe:	8c1d                	sub	s0,s0,a5
1c008300:	47e5                	li	a5,25
1c008302:	0087c563          	blt	a5,s0,1c00830c <__ieee754_rem_pio2f+0x1b0>
1c008306:	c08c                	sw	a1,0(s1)
1c008308:	87b6                	mv	a5,a3
1c00830a:	bf49                	j	1c00829c <__ieee754_rem_pio2f+0x140>
1c00830c:	1c0017b7          	lui	a5,0x1c001
1c008310:	bf47a703          	lw	a4,-1036(a5) # 1c000bf4 <PIo2+0x298>
1c008314:	68e677cb          	fnmsub.s	a5,a2,a4,a3
1c008318:	08f6f6d3          	fsub.s	a3,a3,a5
1c00831c:	68e676cb          	fnmsub.s	a3,a2,a4,a3
1c008320:	1c001737          	lui	a4,0x1c001
1c008324:	bf872703          	lw	a4,-1032(a4) # 1c000bf8 <PIo2+0x29c>
1c008328:	68e67747          	fmsub.s	a4,a2,a4,a3
1c00832c:	08e7f6d3          	fsub.s	a3,a5,a4
1c008330:	c094                	sw	a3,0(s1)
1c008332:	b7ad                	j	1c00829c <__ieee754_rem_pio2f+0x140>
1c008334:	7f800737          	lui	a4,0x7f800
1c008338:	00e44763          	blt	s0,a4,1c008346 <__ieee754_rem_pio2f+0x1ea>
1c00833c:	08a577d3          	fsub.s	a5,a0,a0
1c008340:	c1dc                	sw	a5,4(a1)
1c008342:	c19c                	sw	a5,0(a1)
1c008344:	bd25                	j	1c00817c <__ieee754_rem_pio2f+0x20>
1c008346:	41745613          	srai	a2,s0,0x17
1c00834a:	f7a60613          	addi	a2,a2,-134
1c00834e:	01761793          	slli	a5,a2,0x17
1c008352:	8c1d                	sub	s0,s0,a5
1c008354:	c00417d3          	fcvt.w.s	a5,s0,rtz
1c008358:	00000593          	li	a1,0
1c00835c:	468d                	li	a3,3
1c00835e:	d007f7d3          	fcvt.s.w	a5,a5
1c008362:	08f47453          	fsub.s	s0,s0,a5
1c008366:	c23e                	sw	a5,4(sp)
1c008368:	1c0017b7          	lui	a5,0x1c001
1c00836c:	bfc7a783          	lw	a5,-1028(a5) # 1c000bfc <PIo2+0x2a0>
1c008370:	10f47453          	fmul.s	s0,s0,a5
1c008374:	c0041753          	fcvt.w.s	a4,s0,rtz
1c008378:	d0077753          	fcvt.s.w	a4,a4
1c00837c:	08e47453          	fsub.s	s0,s0,a4
1c008380:	c43a                	sw	a4,8(sp)
1c008382:	10f477d3          	fmul.s	a5,s0,a5
1c008386:	c63e                	sw	a5,12(sp)
1c008388:	a0b7a7d3          	feq.s	a5,a5,a1
1c00838c:	c791                	beqz	a5,1c008398 <__ieee754_rem_pio2f+0x23c>
1c00838e:	a0b72753          	feq.s	a4,a4,a1
1c008392:	00173693          	seqz	a3,a4
1c008396:	0685                	addi	a3,a3,1
1c008398:	1c0007b7          	lui	a5,0x1c000
1c00839c:	63878793          	addi	a5,a5,1592 # 1c000638 <two_over_pi>
1c0083a0:	4709                	li	a4,2
1c0083a2:	85a6                	mv	a1,s1
1c0083a4:	0048                	addi	a0,sp,4
1c0083a6:	28c9                	jal	1c008478 <__kernel_rem_pio2f>
1c0083a8:	e20950e3          	bgez	s2,1c0081c8 <__ieee754_rem_pio2f+0x6c>
1c0083ac:	409c                	lw	a5,0(s1)
1c0083ae:	20f797d3          	fneg.s	a5,a5a5
1c0083b2:	c09c                	sw	a5,0(s1)
1c0083b4:	40dc                	lw	a5,4(s1)
1c0083b6:	bdf5                	j	1c0082b2 <__ieee754_rem_pio2f+0x156>

1c0083b8 <__kernel_cosf>:
1c0083b8:	c1f53633          	p.bclr	a2,a0,0,31
1c0083bc:	320007b7          	lui	a5,0x32000
1c0083c0:	1c001837          	lui	a6,0x1c001
1c0083c4:	00f65563          	ble	a5,a2,1c0083ce <__kernel_cosf+0x16>
1c0083c8:	c00517d3          	fcvt.w.s	a5,a0,rtz
1c0083cc:	c3dd                	beqz	a5,1c008472 <__kernel_cosf+0xba>
1c0083ce:	10a577d3          	fmul.s	a5,a0,a0
1c0083d2:	1c001737          	lui	a4,0x1c001
1c0083d6:	bf072683          	lw	a3,-1040(a4) # 1c000bf0 <PIo2+0x294>
1c0083da:	10b57553          	fmul.s	a0,a0,a1
1c0083de:	1c001737          	lui	a4,0x1c001
1c0083e2:	1c0015b7          	lui	a1,0x1c001
1c0083e6:	c085a583          	lw	a1,-1016(a1) # 1c000c08 <PIo2+0x2ac>
1c0083ea:	c0472703          	lw	a4,-1020(a4) # 1c000c04 <PIo2+0x2a8>
1c0083ee:	10d7f6d3          	fmul.s	a3,a5,a3
1c0083f2:	58e7f743          	fmadd.s	a4,a5,a4,a1
1c0083f6:	1c0015b7          	lui	a1,0x1c001
1c0083fa:	c0c5a583          	lw	a1,-1012(a1) # 1c000c0c <PIo2+0x2b0>
1c0083fe:	58f77743          	fmadd.s	a4,a4,a5,a1
1c008402:	1c0015b7          	lui	a1,0x1c001
1c008406:	c105a583          	lw	a1,-1008(a1) # 1c000c10 <PIo2+0x2b4>
1c00840a:	58f77743          	fmadd.s	a4,a4,a5,a1
1c00840e:	1c0015b7          	lui	a1,0x1c001
1c008412:	c145a583          	lw	a1,-1004(a1) # 1c000c14 <PIo2+0x2b8>
1c008416:	58f77743          	fmadd.s	a4,a4,a5,a1
1c00841a:	1c0015b7          	lui	a1,0x1c001
1c00841e:	c185a583          	lw	a1,-1000(a1) # 1c000c18 <PIo2+0x2bc>
1c008422:	58f77743          	fmadd.s	a4,a4,a5,a1
1c008426:	10f77753          	fmul.s	a4,a4,a5
1c00842a:	50e7f7c7          	fmsub.s	a5,a5,a4,a0
1c00842e:	3e99a737          	lui	a4,0x3e99a
1c008432:	99970713          	addi	a4,a4,-1639 # 3e999999 <__l2_shared_end+0x229827f9>
1c008436:	00c74963          	blt	a4,a2,1c008448 <__kernel_cosf+0x90>
1c00843a:	08f6f7d3          	fsub.s	a5,a3,a5
1c00843e:	98c82503          	lw	a0,-1652(a6) # 1c00098c <PIo2+0x30>
1c008442:	08f57553          	fsub.s	a0,a0,a5
1c008446:	8082                	ret
1c008448:	3f480737          	lui	a4,0x3f480
1c00844c:	00c74e63          	blt	a4,a2,1c008468 <__kernel_cosf+0xb0>
1c008450:	ff000537          	lui	a0,0xff000
1c008454:	962a                	add	a2,a2,a0
1c008456:	98c82503          	lw	a0,-1652(a6)
1c00845a:	08c57553          	fsub.s	a0,a0,a2
1c00845e:	08c6f653          	fsub.s	a2,a3,a2
1c008462:	08f677d3          	fsub.s	a5,a2,a5
1c008466:	bff1                	j	1c008442 <__kernel_cosf+0x8a>
1c008468:	1c001737          	lui	a4,0x1c001
1c00846c:	c0072603          	lw	a2,-1024(a4) # 1c000c00 <PIo2+0x2a4>
1c008470:	b7dd                	j	1c008456 <__kernel_cosf+0x9e>
1c008472:	98c82503          	lw	a0,-1652(a6)
1c008476:	8082                	ret

1c008478 <__kernel_rem_pio2f>:
1c008478:	7125                	addi	sp,sp,-416
1c00847a:	19512223          	sw	s5,388(sp)
1c00847e:	1c001ab7          	lui	s5,0x1c001
1c008482:	19212823          	sw	s2,400(sp)
1c008486:	19312623          	sw	s3,396(sp)
1c00848a:	950a8913          	addi	s2,s5,-1712 # 1c000950 <init_jk>
1c00848e:	89ba                	mv	s3,a4
1c008490:	070a                	slli	a4,a4,0x2
1c008492:	20e97903          	p.lw	s2,a4(s2)
1c008496:	ffd60e13          	addi	t3,a2,-3
1c00849a:	4721                	li	a4,8
1c00849c:	02ee4e33          	div	t3,t3,a4
1c0084a0:	19612023          	sw	s6,384(sp)
1c0084a4:	17712e23          	sw	s7,380(sp)
1c0084a8:	fff68b93          	addi	s7,a3,-1
1c0084ac:	17912a23          	sw	s9,372(sp)
1c0084b0:	cc2a                	sw	a0,24(sp)
1c0084b2:	8cae                	mv	s9,a1
1c0084b4:	188c                	addi	a1,sp,112
1c0084b6:	18912a23          	sw	s1,404(sp)
1c0084ba:	18112e23          	sw	ra,412(sp)
1c0084be:	18812c23          	sw	s0,408(sp)
1c0084c2:	19412423          	sw	s4,392(sp)
1c0084c6:	17812c23          	sw	s8,376(sp)
1c0084ca:	17a12823          	sw	s10,368(sp)
1c0084ce:	17b12623          	sw	s11,364(sp)
1c0084d2:	01790f33          	add	t5,s2,s7
1c0084d6:	8aae                	mv	s5,a1
1c0084d8:	040e6b33          	p.max	s6,t3,zero
1c0084dc:	001b0713          	addi	a4,s6,1
1c0084e0:	417b0333          	sub	t1,s6,s7
1c0084e4:	070e                	slli	a4,a4,0x3
1c0084e6:	00231513          	slli	a0,t1,0x2
1c0084ea:	40e604b3          	sub	s1,a2,a4
1c0084ee:	953e                	add	a0,a0,a5
1c0084f0:	4601                	li	a2,0
1c0084f2:	02cf5063          	ble	a2,t5,1c008512 <__kernel_rem_pio2f+0x9a>
1c0084f6:	00269713          	slli	a4,a3,0x2
1c0084fa:	9756                	add	a4,a4,s5
1c0084fc:	11010313          	addi	t1,sp,272
1c008500:	4501                	li	a0,0
1c008502:	5f71                	li	t5,-4
1c008504:	04a94863          	blt	s2,a0,1c008554 <__kernel_rem_pio2f+0xdc>
1c008508:	4862                	lw	a6,24(sp)
1c00850a:	00000593          	li	a1,0
1c00850e:	4601                	li	a2,0
1c008510:	a81d                	j	1c008546 <__kernel_rem_pio2f+0xce>
1c008512:	00c30833          	add	a6,t1,a2
1c008516:	00000713          	li	a4,0
1c00851a:	00084863          	bltz	a6,1c00852a <__kernel_rem_pio2f+0xb2>
1c00851e:	00261713          	slli	a4,a2,0x2
1c008522:	20e57703          	p.lw	a4,a4(a0)
1c008526:	d0077753          	fcvt.s.w	a4,a4
1c00852a:	00e5a22b          	p.sw	a4,4(a1!)
1c00852e:	0605                	addi	a2,a2,1
1c008530:	b7c9                	j	1c0084f2 <__kernel_rem_pio2f+0x7a>
1c008532:	8fba                	mv	t6,a4
1c008534:	43e60fb3          	p.mac	t6,a2,t5
1c008538:	0048228b          	p.lw	t0,4(a6!)
1c00853c:	0605                	addi	a2,a2,1
1c00853e:	ffcfaf83          	lw	t6,-4(t6)
1c008542:	59f2f5c3          	fmadd.s	a1,t0,t6,a1
1c008546:	fecbd6e3          	ble	a2,s7,1c008532 <__kernel_rem_pio2f+0xba>
1c00854a:	00b3222b          	p.sw	a1,4(t1!)
1c00854e:	0505                	addi	a0,a0,1
1c008550:	0711                	addi	a4,a4,4
1c008552:	bf4d                	j	1c008504 <__kernel_rem_pio2f+0x8c>
1c008554:	00291a13          	slli	s4,s2,0x2
1c008558:	1a71                	addi	s4,s4,-4
1c00855a:	1010                	addi	a2,sp,32
1c00855c:	9652                	add	a2,a2,s4
1c00855e:	ce32                	sw	a2,28(sp)
1c008560:	8dca                	mv	s11,s2
1c008562:	002d9a13          	slli	s4,s11,0x2
1c008566:	1290                	addi	a2,sp,352
1c008568:	01460733          	add	a4,a2,s4
1c00856c:	fb072503          	lw	a0,-80(a4)
1c008570:	02010f13          	addi	t5,sp,32
1c008574:	0a18                	addi	a4,sp,272
1c008576:	01470333          	add	t1,a4,s4
1c00857a:	8ffa                	mv	t6,t5
1c00857c:	85ee                	mv	a1,s11
1c00857e:	1371                	addi	t1,t1,-4
1c008580:	14b04163          	bgtz	a1,1c0086c2 <__kernel_rem_pio2f+0x24a>
1c008584:	85a6                	mv	a1,s1
1c008586:	c836                	sw	a3,16(sp)
1c008588:	c63e                	sw	a5,12(sp)
1c00858a:	ca7a                	sw	t5,20(sp)
1c00858c:	2b95                	jal	1c008b00 <scalbnf>
1c00858e:	1c001737          	lui	a4,0x1c001
1c008592:	c2070713          	addi	a4,a4,-992 # 1c000c20 <PIo2+0x2c4>
1c008596:	4318                	lw	a4,0(a4)
1c008598:	842a                	mv	s0,a0
1c00859a:	10e57553          	fmul.s	a0,a0,a4
1c00859e:	29e9                	jal	1c008a78 <floorf>
1c0085a0:	1c001637          	lui	a2,0x1c001
1c0085a4:	c2460613          	addi	a2,a2,-988 # 1c000c24 <PIo2+0x2c8>
1c0085a8:	4210                	lw	a2,0(a2)
1c0085aa:	46c2                	lw	a3,16(sp)
1c0085ac:	47b2                	lw	a5,12(sp)
1c0085ae:	40c5744b          	fnmsub.s	s0,a0,a2,s0
1c0085b2:	4f52                	lw	t5,20(sp)
1c0085b4:	c0041c53          	fcvt.w.s	s8,s0,rtz
1c0085b8:	d00c7553          	fcvt.s.w	a0,s8
1c0085bc:	08a47453          	fsub.s	s0,s0,a0
1c0085c0:	12905d63          	blez	s1,1c0086fa <__kernel_rem_pio2f+0x282>
1c0085c4:	fffd8593          	addi	a1,s11,-1
1c0085c8:	1298                	addi	a4,sp,352
1c0085ca:	058a                	slli	a1,a1,0x2
1c0085cc:	95ba                	add	a1,a1,a4
1c0085ce:	ec05a703          	lw	a4,-320(a1)
1c0085d2:	4521                	li	a0,8
1c0085d4:	40950fb3          	sub	t6,a0,s1
1c0085d8:	41f75533          	sra	a0,a4,t6
1c0085dc:	9c2a                	add	s8,s8,a0
1c0085de:	01f51533          	sll	a0,a0,t6
1c0085e2:	8f09                	sub	a4,a4,a0
1c0085e4:	ece5a023          	sw	a4,-320(a1)
1c0085e8:	459d                	li	a1,7
1c0085ea:	8d85                	sub	a1,a1,s1
1c0085ec:	40b75d33          	sra	s10,a4,a1
1c0085f0:	05a05363          	blez	s10,1c008636 <__kernel_rem_pio2f+0x1be>
1c0085f4:	0c05                	addi	s8,s8,1
1c0085f6:	4501                	li	a0,0
1c0085f8:	4381                	li	t2,0
1c0085fa:	0ff00f93          	li	t6,255
1c0085fe:	10000293          	li	t0,256
1c008602:	13b54163          	blt	a0,s11,1c008724 <__kernel_rem_pio2f+0x2ac>
1c008606:	00905663          	blez	s1,1c008612 <__kernel_rem_pio2f+0x19a>
1c00860a:	1414a163          	p.beqimm	s1,1,1c00874c <__kernel_rem_pio2f+0x2d4>
1c00860e:	1424ab63          	p.beqimm	s1,2,1c008764 <__kernel_rem_pio2f+0x2ec>
1c008612:	022d3263          	p.bneimm	s10,2,1c008636 <__kernel_rem_pio2f+0x1be>
1c008616:	1c0015b7          	lui	a1,0x1c001
1c00861a:	98c5a503          	lw	a0,-1652(a1) # 1c00098c <PIo2+0x30>
1c00861e:	08857453          	fsub.s	s0,a0,s0
1c008622:	00038a63          	beqz	t2,1c008636 <__kernel_rem_pio2f+0x1be>
1c008626:	85a6                	mv	a1,s1
1c008628:	c836                	sw	a3,16(sp)
1c00862a:	c63e                	sw	a5,12(sp)
1c00862c:	29d1                	jal	1c008b00 <scalbnf>
1c00862e:	08a47453          	fsub.s	s0,s0,a0
1c008632:	47b2                	lw	a5,12(sp)
1c008634:	46c2                	lw	a3,16(sp)
1c008636:	00000593          	li	a1,0
1c00863a:	a0b425d3          	feq.s	a1,s0,a1
1c00863e:	1a058b63          	beqz	a1,1c0087f4 <__kernel_rem_pio2f+0x37c>
1c008642:	fffd8413          	addi	s0,s11,-1
1c008646:	1018                	addi	a4,sp,32
1c008648:	01470f33          	add	t5,a4,s4
1c00864c:	8522                	mv	a0,s0
1c00864e:	4581                	li	a1,0
1c008650:	1f71                	addi	t5,t5,-4
1c008652:	13255363          	ble	s2,a0,1c008778 <__kernel_rem_pio2f+0x300>
1c008656:	14058f63          	beqz	a1,1c0087b4 <__kernel_rem_pio2f+0x33c>
1c00865a:	00241793          	slli	a5,s0,0x2
1c00865e:	1010                	addi	a2,sp,32
1c008660:	14e1                	addi	s1,s1,-8
1c008662:	97b2                	add	a5,a5,a2
1c008664:	ffc7a68b          	p.lw	a3,-4(a5!) # 31fffffc <__l2_shared_end+0x15fe8e5c>
1c008668:	18068363          	beqz	a3,1c0087ee <__kernel_rem_pio2f+0x376>
1c00866c:	1c0017b7          	lui	a5,0x1c001
1c008670:	98c7a503          	lw	a0,-1652(a5) # 1c00098c <PIo2+0x30>
1c008674:	85a6                	mv	a1,s1
1c008676:	2169                	jal	1c008b00 <scalbnf>
1c008678:	00241793          	slli	a5,s0,0x2
1c00867c:	1010                	addi	a2,sp,32
1c00867e:	00f606b3          	add	a3,a2,a5
1c008682:	1c001637          	lui	a2,0x1c001
1c008686:	c1c62e03          	lw	t3,-996(a2) # 1c000c1c <PIo2+0x2c0>
1c00868a:	0a18                	addi	a4,sp,272
1c00868c:	00f70833          	add	a6,a4,a5
1c008690:	85a2                	mv	a1,s0
1c008692:	1c05d363          	bgez	a1,1c008858 <__kernel_rem_pio2f+0x3e0>
1c008696:	0a10                	addi	a2,sp,272
1c008698:	00c786b3          	add	a3,a5,a2
1c00869c:	4601                	li	a2,0
1c00869e:	40c405b3          	sub	a1,s0,a2
1c0086a2:	1e05c863          	bltz	a1,1c008892 <__kernel_rem_pio2f+0x41a>
1c0086a6:	1c001737          	lui	a4,0x1c001
1c0086aa:	95070713          	addi	a4,a4,-1712 # 1c000950 <init_jk>
1c0086ae:	00261513          	slli	a0,a2,0x2
1c0086b2:	00c70e93          	addi	t4,a4,12
1c0086b6:	40a68e33          	sub	t3,a3,a0
1c0086ba:	00000813          	li	a6,0
1c0086be:	4581                	li	a1,0
1c0086c0:	aa7d                	j	1c00887e <__kernel_rem_pio2f+0x406>
1c0086c2:	1c001637          	lui	a2,0x1c001
1c0086c6:	c1c60613          	addi	a2,a2,-996 # 1c000c1c <PIo2+0x2c0>
1c0086ca:	4210                	lw	a2,0(a2)
1c0086cc:	15fd                	addi	a1,a1,-1
1c0086ce:	10c57753          	fmul.s	a4,a0,a2
1c0086d2:	1c001637          	lui	a2,0x1c001
1c0086d6:	bfc60613          	addi	a2,a2,-1028 # 1c000bfc <PIo2+0x2a0>
1c0086da:	4210                	lw	a2,0(a2)
1c0086dc:	c0071753          	fcvt.w.s	a4,a4,rtz
1c0086e0:	d0077753          	fcvt.s.w	a4,a4
1c0086e4:	50c7754b          	fnmsub.s	a0,a4,a2,a0
1c0086e8:	c0051553          	fcvt.w.s	a0,a0,rtz
1c0086ec:	00afa22b          	p.sw	a0,4(t6!)
1c0086f0:	00032503          	lw	a0,0(t1)
1c0086f4:	00a77553          	fadd.s	a0,a4,a0
1c0086f8:	b559                	j	1c00857e <__kernel_rem_pio2f+0x106>
1c0086fa:	e899                	bnez	s1,1c008710 <__kernel_rem_pio2f+0x298>
1c0086fc:	fffd8713          	addi	a4,s11,-1
1c008700:	070a                	slli	a4,a4,0x2
1c008702:	1290                	addi	a2,sp,352
1c008704:	9732                	add	a4,a4,a2
1c008706:	ec072703          	lw	a4,-320(a4)
1c00870a:	40875d13          	srai	s10,a4,0x8
1c00870e:	b5cd                	j	1c0085f0 <__kernel_rem_pio2f+0x178>
1c008710:	1c001737          	lui	a4,0x1c001
1c008714:	bf072703          	lw	a4,-1040(a4) # 1c000bf0 <PIo2+0x294>
1c008718:	4d01                	li	s10,0
1c00871a:	a0870753          	fle.s	a4,a4,s0
1c00871e:	df01                	beqz	a4,1c008636 <__kernel_rem_pio2f+0x1be>
1c008720:	4d09                	li	s10,2
1c008722:	bdc9                	j	1c0085f4 <__kernel_rem_pio2f+0x17c>
1c008724:	000f2583          	lw	a1,0(t5)
1c008728:	00039c63          	bnez	t2,1c008740 <__kernel_rem_pio2f+0x2c8>
1c00872c:	c591                	beqz	a1,1c008738 <__kernel_rem_pio2f+0x2c0>
1c00872e:	40b285b3          	sub	a1,t0,a1
1c008732:	00bf2023          	sw	a1,0(t5)
1c008736:	4585                	li	a1,1
1c008738:	0505                	addi	a0,a0,1
1c00873a:	0f11                	addi	t5,t5,4
1c00873c:	83ae                	mv	t2,a1
1c00873e:	b5d1                	j	1c008602 <__kernel_rem_pio2f+0x18a>
1c008740:	40bf85b3          	sub	a1,t6,a1
1c008744:	00bf2023          	sw	a1,0(t5)
1c008748:	859e                	mv	a1,t2
1c00874a:	b7fd                	j	1c008738 <__kernel_rem_pio2f+0x2c0>
1c00874c:	fffd8593          	addi	a1,s11,-1
1c008750:	058a                	slli	a1,a1,0x2
1c008752:	1298                	addi	a4,sp,352
1c008754:	95ba                	add	a1,a1,a4
1c008756:	ec05a503          	lw	a0,-320(a1)
1c00875a:	f0753533          	p.bclr	a0,a0,24,7
1c00875e:	eca5a023          	sw	a0,-320(a1)
1c008762:	bd45                	j	1c008612 <__kernel_rem_pio2f+0x19a>
1c008764:	fffd8593          	addi	a1,s11,-1
1c008768:	058a                	slli	a1,a1,0x2
1c00876a:	1290                	addi	a2,sp,352
1c00876c:	95b2                	add	a1,a1,a2
1c00876e:	ec05a503          	lw	a0,-320(a1)
1c008772:	f2653533          	p.bclr	a0,a0,25,6
1c008776:	b7e5                	j	1c00875e <__kernel_rem_pio2f+0x2e6>
1c008778:	000f2f83          	lw	t6,0(t5)
1c00877c:	157d                	addi	a0,a0,-1
1c00877e:	01f5e5b3          	or	a1,a1,t6
1c008782:	b5f9                	j	1c008650 <__kernel_rem_pio2f+0x1d8>
1c008784:	0585                	addi	a1,a1,1
1c008786:	ffc7250b          	p.lw	a0,-4(a4!)
1c00878a:	dd6d                	beqz	a0,1c008784 <__kernel_rem_pio2f+0x30c>
1c00878c:	0a18                	addi	a4,sp,272
1c00878e:	001d8f93          	addi	t6,s11,1
1c008792:	004a0613          	addi	a2,s4,4
1c008796:	963a                	add	a2,a2,a4
1c008798:	01fb0f33          	add	t5,s6,t6
1c00879c:	00dd8733          	add	a4,s11,a3
1c0087a0:	0f0a                	slli	t5,t5,0x2
1c0087a2:	070a                	slli	a4,a4,0x2
1c0087a4:	9f3e                	add	t5,t5,a5
1c0087a6:	9756                	add	a4,a4,s5
1c0087a8:	00bd8833          	add	a6,s11,a1
1c0087ac:	01f85763          	ble	t6,a6,1c0087ba <__kernel_rem_pio2f+0x342>
1c0087b0:	8dc2                	mv	s11,a6
1c0087b2:	bb45                	j	1c008562 <__kernel_rem_pio2f+0xea>
1c0087b4:	4772                	lw	a4,28(sp)
1c0087b6:	4585                	li	a1,1
1c0087b8:	b7f9                	j	1c008786 <__kernel_rem_pio2f+0x30e>
1c0087ba:	004f250b          	p.lw	a0,4(t5!)
1c0087be:	85ba                	mv	a1,a4
1c0087c0:	42e2                	lw	t0,24(sp)
1c0087c2:	d0057553          	fcvt.s.w	a0,a0
1c0087c6:	4301                	li	t1,0
1c0087c8:	00a5a22b          	p.sw	a0,4(a1!)
1c0087cc:	00000513          	li	a0,0
1c0087d0:	006bd763          	ble	t1,s7,1c0087de <__kernel_rem_pio2f+0x366>
1c0087d4:	00a6222b          	p.sw	a0,4(a2!)
1c0087d8:	0f85                	addi	t6,t6,1
1c0087da:	872e                	mv	a4,a1
1c0087dc:	bfc1                	j	1c0087ac <__kernel_rem_pio2f+0x334>
1c0087de:	0042a40b          	p.lw	s0,4(t0!)
1c0087e2:	ffc7238b          	p.lw	t2,-4(a4!)
1c0087e6:	0305                	addi	t1,t1,1
1c0087e8:	50747543          	fmadd.s	a0,s0,t2,a0
1c0087ec:	b7d5                	j	1c0087d0 <__kernel_rem_pio2f+0x358>
1c0087ee:	147d                	addi	s0,s0,-1
1c0087f0:	14e1                	addi	s1,s1,-8
1c0087f2:	bd8d                	j	1c008664 <__kernel_rem_pio2f+0x1ec>
1c0087f4:	8522                	mv	a0,s0
1c0087f6:	409005b3          	neg	a1,s1
1c0087fa:	2619                	jal	1c008b00 <scalbnf>
1c0087fc:	1c0017b7          	lui	a5,0x1c001
1c008800:	bfc7a683          	lw	a3,-1028(a5) # 1c000bfc <PIo2+0x2a0>
1c008804:	a0a687d3          	fle.s	a5,a3,a0
1c008808:	cf9d                	beqz	a5,1c008846 <__kernel_rem_pio2f+0x3ce>
1c00880a:	1c0017b7          	lui	a5,0x1c001
1c00880e:	c1c7a783          	lw	a5,-996(a5) # 1c000c1c <PIo2+0x2c0>
1c008812:	1298                	addi	a4,sp,352
1c008814:	001d8413          	addi	s0,s11,1
1c008818:	10f577d3          	fmul.s	a5,a0,a5
1c00881c:	01470633          	add	a2,a4,s4
1c008820:	04a1                	addi	s1,s1,8
1c008822:	c00797d3          	fcvt.w.s	a5,a5,rtz
1c008826:	d007f7d3          	fcvt.s.w	a5,a5
1c00882a:	50d7f54b          	fnmsub.s	a0,a5,a3,a0
1c00882e:	c00797d3          	fcvt.w.s	a5,a5,rtz
1c008832:	00241693          	slli	a3,s0,0x2
1c008836:	96ba                	add	a3,a3,a4
1c008838:	c0051553          	fcvt.w.s	a0,a0,rtz
1c00883c:	eca62023          	sw	a0,-320(a2)
1c008840:	ecf6a023          	sw	a5,-320(a3)
1c008844:	b525                	j	1c00866c <__kernel_rem_pio2f+0x1f4>
1c008846:	c0051553          	fcvt.w.s	a0,a0,rtz
1c00884a:	129c                	addi	a5,sp,352
1c00884c:	01478633          	add	a2,a5,s4
1c008850:	eca62023          	sw	a0,-320(a2)
1c008854:	846e                	mv	s0,s11
1c008856:	bd19                	j	1c00866c <__kernel_rem_pio2f+0x1f4>
1c008858:	ffc6a60b          	p.lw	a2,-4(a3!)
1c00885c:	15fd                	addi	a1,a1,-1
1c00885e:	d0067653          	fcvt.s.w	a2,a2
1c008862:	10a67653          	fmul.s	a2,a2,a0
1c008866:	11c57553          	fmul.s	a0,a0,t3
1c00886a:	fec82e2b          	p.sw	a2,-4(a6!)
1c00886e:	b515                	j	1c008692 <__kernel_rem_pio2f+0x21a>
1c008870:	004eaf8b          	p.lw	t6,4(t4!)
1c008874:	004e2f0b          	p.lw	t5,4(t3!)
1c008878:	0585                	addi	a1,a1,1
1c00887a:	81eff843          	fmadd.s	a6,t6,t5,a6
1c00887e:	00b94463          	blt	s2,a1,1c008886 <__kernel_rem_pio2f+0x40e>
1c008882:	feb657e3          	ble	a1,a2,1c008870 <__kernel_rem_pio2f+0x3f8>
1c008886:	1298                	addi	a4,sp,352
1c008888:	953a                	add	a0,a0,a4
1c00888a:	f7052023          	sw	a6,-160(a0) # feffff60 <pulp__FC+0xfeffff61>
1c00888e:	0605                	addi	a2,a2,1
1c008890:	b539                	j	1c00869e <__kernel_rem_pio2f+0x226>
1c008892:	4689                	li	a3,2
1c008894:	0536c463          	blt	a3,s3,1c0088dc <__kernel_rem_pio2f+0x464>
1c008898:	09304163          	bgtz	s3,1c00891a <__kernel_rem_pio2f+0x4a2>
1c00889c:	0a098c63          	beqz	s3,1c008954 <__kernel_rem_pio2f+0x4dc>
1c0088a0:	19c12083          	lw	ra,412(sp)
1c0088a4:	19812403          	lw	s0,408(sp)
1c0088a8:	f83c3533          	p.bclr	a0,s8,28,3
1c0088ac:	19412483          	lw	s1,404(sp)
1c0088b0:	19012903          	lw	s2,400(sp)
1c0088b4:	18c12983          	lw	s3,396(sp)
1c0088b8:	18812a03          	lw	s4,392(sp)
1c0088bc:	18412a83          	lw	s5,388(sp)
1c0088c0:	18012b03          	lw	s6,384(sp)
1c0088c4:	17c12b83          	lw	s7,380(sp)
1c0088c8:	17812c03          	lw	s8,376(sp)
1c0088cc:	17412c83          	lw	s9,372(sp)
1c0088d0:	17012d03          	lw	s10,368(sp)
1c0088d4:	16c12d83          	lw	s11,364(sp)
1c0088d8:	611d                	addi	sp,sp,416
1c0088da:	8082                	ret
1c0088dc:	fc39b2e3          	p.bneimm	s3,3,1c0088a0 <__kernel_rem_pio2f+0x428>
1c0088e0:	0190                	addi	a2,sp,192
1c0088e2:	97b2                	add	a5,a5,a2
1c0088e4:	86be                	mv	a3,a5
1c0088e6:	85a2                	mv	a1,s0
1c0088e8:	16f1                	addi	a3,a3,-4
1c0088ea:	0ab04363          	bgtz	a1,1c008990 <__kernel_rem_pio2f+0x518>
1c0088ee:	86be                	mv	a3,a5
1c0088f0:	85a2                	mv	a1,s0
1c0088f2:	4e05                	li	t3,1
1c0088f4:	16f1                	addi	a3,a3,-4
1c0088f6:	0abe4a63          	blt	t3,a1,1c0089aa <__kernel_rem_pio2f+0x532>
1c0088fa:	00000693          	li	a3,0
1c0088fe:	4605                	li	a2,1
1c008900:	0c864263          	blt	a2,s0,1c0089c4 <__kernel_rem_pio2f+0x54c>
1c008904:	460e                	lw	a2,192(sp)
1c008906:	479e                	lw	a5,196(sp)
1c008908:	0c0d1463          	bnez	s10,1c0089d0 <__kernel_rem_pio2f+0x558>
1c00890c:	00cca023          	sw	a2,0(s9)
1c008910:	00fca223          	sw	a5,4(s9)
1c008914:	00dca423          	sw	a3,8(s9)
1c008918:	b761                	j	1c0088a0 <__kernel_rem_pio2f+0x428>
1c00891a:	00000693          	li	a3,0
1c00891e:	0198                	addi	a4,sp,192
1c008920:	97ba                	add	a5,a5,a4
1c008922:	8622                	mv	a2,s0
1c008924:	04065b63          	bgez	a2,1c00897a <__kernel_rem_pio2f+0x502>
1c008928:	87b6                	mv	a5,a3
1c00892a:	000d0463          	beqz	s10,1c008932 <__kernel_rem_pio2f+0x4ba>
1c00892e:	20d697d3          	fneg.s	a5,a3a3
1c008932:	00fca023          	sw	a5,0(s9)
1c008936:	478e                	lw	a5,192(sp)
1c008938:	4605                	li	a2,1
1c00893a:	08d7f7d3          	fsub.s	a5,a5,a3
1c00893e:	0194                	addi	a3,sp,192
1c008940:	0691                	addi	a3,a3,4
1c008942:	04c45263          	ble	a2,s0,1c008986 <__kernel_rem_pio2f+0x50e>
1c008946:	000d0463          	beqz	s10,1c00894e <__kernel_rem_pio2f+0x4d6>
1c00894a:	20f797d3          	fneg.s	a5,a5a5
1c00894e:	00fca223          	sw	a5,4(s9)
1c008952:	b7b9                	j	1c0088a0 <__kernel_rem_pio2f+0x428>
1c008954:	00000693          	li	a3,0
1c008958:	0190                	addi	a2,sp,192
1c00895a:	97b2                	add	a5,a5,a2
1c00895c:	00045963          	bgez	s0,1c00896e <__kernel_rem_pio2f+0x4f6>
1c008960:	000d0463          	beqz	s10,1c008968 <__kernel_rem_pio2f+0x4f0>
1c008964:	20d696d3          	fneg.s	a3,a3a3
1c008968:	00dca023          	sw	a3,0(s9)
1c00896c:	bf15                	j	1c0088a0 <__kernel_rem_pio2f+0x428>
1c00896e:	ffc7a60b          	p.lw	a2,-4(a5!)
1c008972:	147d                	addi	s0,s0,-1
1c008974:	00c6f6d3          	fadd.s	a3,a3,a2
1c008978:	b7d5                	j	1c00895c <__kernel_rem_pio2f+0x4e4>
1c00897a:	ffc7a58b          	p.lw	a1,-4(a5!)
1c00897e:	167d                	addi	a2,a2,-1
1c008980:	00b6f6d3          	fadd.s	a3,a3,a1
1c008984:	b745                	j	1c008924 <__kernel_rem_pio2f+0x4ac>
1c008986:	428c                	lw	a1,0(a3)
1c008988:	0605                	addi	a2,a2,1
1c00898a:	00b7f7d3          	fadd.s	a5,a5,a1
1c00898e:	bf4d                	j	1c008940 <__kernel_rem_pio2f+0x4c8>
1c008990:	4290                	lw	a2,0(a3)
1c008992:	0046a803          	lw	a6,4(a3)
1c008996:	15fd                	addi	a1,a1,-1
1c008998:	01067553          	fadd.s	a0,a2,a6
1c00899c:	08a67653          	fsub.s	a2,a2,a0
1c0089a0:	c288                	sw	a0,0(a3)
1c0089a2:	01067653          	fadd.s	a2,a2,a6
1c0089a6:	c2d0                	sw	a2,4(a3)
1c0089a8:	b781                	j	1c0088e8 <__kernel_rem_pio2f+0x470>
1c0089aa:	4290                	lw	a2,0(a3)
1c0089ac:	0046a803          	lw	a6,4(a3)
1c0089b0:	15fd                	addi	a1,a1,-1
1c0089b2:	01067553          	fadd.s	a0,a2,a6
1c0089b6:	08a67653          	fsub.s	a2,a2,a0
1c0089ba:	c288                	sw	a0,0(a3)
1c0089bc:	01067653          	fadd.s	a2,a2,a6
1c0089c0:	c2d0                	sw	a2,4(a3)
1c0089c2:	bf0d                	j	1c0088f4 <__kernel_rem_pio2f+0x47c>
1c0089c4:	ffc7a58b          	p.lw	a1,-4(a5!)
1c0089c8:	147d                	addi	s0,s0,-1
1c0089ca:	00b6f6d3          	fadd.s	a3,a3,a1
1c0089ce:	bf0d                	j	1c008900 <__kernel_rem_pio2f+0x488>
1c0089d0:	20c61653          	fneg.s	a2,a2a2
1c0089d4:	20f797d3          	fneg.s	a5,a5a5
1c0089d8:	20d696d3          	fneg.s	a3,a3a3
1c0089dc:	00cca023          	sw	a2,0(s9)
1c0089e0:	00fca223          	sw	a5,4(s9)
1c0089e4:	bf05                	j	1c008914 <__kernel_rem_pio2f+0x49c>

1c0089e6 <__kernel_sinf>:
1c0089e6:	c1f53733          	p.bclr	a4,a0,0,31
1c0089ea:	320007b7          	lui	a5,0x32000
1c0089ee:	00f75563          	ble	a5,a4,1c0089f8 <__kernel_sinf+0x12>
1c0089f2:	c00517d3          	fcvt.w.s	a5,a0,rtz
1c0089f6:	cfad                	beqz	a5,1c008a70 <__kernel_sinf+0x8a>
1c0089f8:	10a57753          	fmul.s	a4,a0,a0
1c0089fc:	1c0017b7          	lui	a5,0x1c001
1c008a00:	1c001837          	lui	a6,0x1c001
1c008a04:	c2c82803          	lw	a6,-980(a6) # 1c000c2c <PIo2+0x2d0>
1c008a08:	c287a783          	lw	a5,-984(a5) # 1c000c28 <PIo2+0x2cc>
1c008a0c:	10e576d3          	fmul.s	a3,a0,a4
1c008a10:	80f777c3          	fmadd.s	a5,a4,a5,a6
1c008a14:	1c001837          	lui	a6,0x1c001
1c008a18:	c3082803          	lw	a6,-976(a6) # 1c000c30 <PIo2+0x2d4>
1c008a1c:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a20:	1c001837          	lui	a6,0x1c001
1c008a24:	c3482803          	lw	a6,-972(a6) # 1c000c34 <PIo2+0x2d8>
1c008a28:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a2c:	1c001837          	lui	a6,0x1c001
1c008a30:	c3882803          	lw	a6,-968(a6) # 1c000c38 <PIo2+0x2dc>
1c008a34:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a38:	ea11                	bnez	a2,1c008a4c <__kernel_sinf+0x66>
1c008a3a:	1c001637          	lui	a2,0x1c001
1c008a3e:	c3c62583          	lw	a1,-964(a2) # 1c000c3c <PIo2+0x2e0>
1c008a42:	58f777c3          	fmadd.s	a5,a4,a5,a1
1c008a46:	50d7f543          	fmadd.s	a0,a5,a3,a0
1c008a4a:	8082                	ret
1c008a4c:	10f6f7d3          	fmul.s	a5,a3,a5
1c008a50:	1c001637          	lui	a2,0x1c001
1c008a54:	bf062603          	lw	a2,-1040(a2) # 1c000bf0 <PIo2+0x294>
1c008a58:	78c5f7c7          	fmsub.s	a5,a1,a2,a5
1c008a5c:	58e7f747          	fmsub.s	a4,a5,a4,a1
1c008a60:	1c0017b7          	lui	a5,0x1c001
1c008a64:	c407a783          	lw	a5,-960(a5) # 1c000c40 <PIo2+0x2e4>
1c008a68:	70f6f6c3          	fmadd.s	a3,a3,a5,a4
1c008a6c:	08d57553          	fsub.s	a0,a0,a3
1c008a70:	8082                	ret

1c008a72 <fabsf>:
1c008a72:	c1f53533          	p.bclr	a0,a0,0,31
1c008a76:	8082                	ret

1c008a78 <floorf>:
1c008a78:	c1f53733          	p.bclr	a4,a0,0,31
1c008a7c:	01775693          	srli	a3,a4,0x17
1c008a80:	f8168693          	addi	a3,a3,-127
1c008a84:	47d9                	li	a5,22
1c008a86:	06d7c463          	blt	a5,a3,1c008aee <floorf+0x76>
1c008a8a:	87aa                	mv	a5,a0
1c008a8c:	0206d463          	bgez	a3,1c008ab4 <floorf+0x3c>
1c008a90:	1c0016b7          	lui	a3,0x1c001
1c008a94:	c446a683          	lw	a3,-956(a3) # 1c000c44 <PIo2+0x2e8>
1c008a98:	00d57553          	fadd.s	a0,a0,a3
1c008a9c:	00000693          	li	a3,0
1c008aa0:	a0a69553          	flt.s	a0,a3,a0
1c008aa4:	c511                	beqz	a0,1c008ab0 <floorf+0x38>
1c008aa6:	0407db63          	bgez	a5,1c008afc <floorf+0x84>
1c008aaa:	c319                	beqz	a4,1c008ab0 <floorf+0x38>
1c008aac:	bf8007b7          	lui	a5,0xbf800
1c008ab0:	853e                	mv	a0,a5
1c008ab2:	8082                	ret
1c008ab4:	00800637          	lui	a2,0x800
1c008ab8:	fff60713          	addi	a4,a2,-1 # 7fffff <__l1_heap_size+0x7e7017>
1c008abc:	40d75733          	sra	a4,a4,a3
1c008ac0:	00a775b3          	and	a1,a4,a0
1c008ac4:	d5fd                	beqz	a1,1c008ab2 <floorf+0x3a>
1c008ac6:	1c0015b7          	lui	a1,0x1c001
1c008aca:	c445a583          	lw	a1,-956(a1) # 1c000c44 <PIo2+0x2e8>
1c008ace:	00b57553          	fadd.s	a0,a0,a1
1c008ad2:	00000593          	li	a1,0
1c008ad6:	a0a59553          	flt.s	a0,a1,a0
1c008ada:	d979                	beqz	a0,1c008ab0 <floorf+0x38>
1c008adc:	0007d563          	bgez	a5,1c008ae6 <floorf+0x6e>
1c008ae0:	40d656b3          	sra	a3,a2,a3
1c008ae4:	97b6                	add	a5,a5,a3
1c008ae6:	fff74713          	not	a4,a4
1c008aea:	8ff9                	and	a5,a5,a4
1c008aec:	b7d1                	j	1c008ab0 <floorf+0x38>
1c008aee:	7f8007b7          	lui	a5,0x7f800
1c008af2:	fcf760e3          	bltu	a4,a5,1c008ab2 <floorf+0x3a>
1c008af6:	00a57553          	fadd.s	a0,a0,a0
1c008afa:	8082                	ret
1c008afc:	4781                	li	a5,0
1c008afe:	bf4d                	j	1c008ab0 <floorf+0x38>

1c008b00 <scalbnf>:
1c008b00:	c1f537b3          	p.bclr	a5,a0,0,31
1c008b04:	cfcd                	beqz	a5,1c008bbe <scalbnf+0xbe>
1c008b06:	7f800737          	lui	a4,0x7f800
1c008b0a:	00e7e563          	bltu	a5,a4,1c008b14 <scalbnf+0x14>
1c008b0e:	00a57553          	fadd.s	a0,a0,a0
1c008b12:	8082                	ret
1c008b14:	00800737          	lui	a4,0x800
1c008b18:	04e7fc63          	bleu	a4,a5,1c008b70 <scalbnf+0x70>
1c008b1c:	1c0017b7          	lui	a5,0x1c001
1c008b20:	c487a783          	lw	a5,-952(a5) # 1c000c48 <PIo2+0x2ec>
1c008b24:	10f57553          	fmul.s	a0,a0,a5
1c008b28:	77d1                	lui	a5,0xffff4
1c008b2a:	cb078793          	addi	a5,a5,-848 # ffff3cb0 <pulp__FC+0xffff3cb1>
1c008b2e:	02f5ca63          	blt	a1,a5,1c008b62 <scalbnf+0x62>
1c008b32:	41755793          	srai	a5,a0,0x17
1c008b36:	ee87b7b3          	p.bclr	a5,a5,23,8
1c008b3a:	872a                	mv	a4,a0
1c008b3c:	179d                	addi	a5,a5,-25
1c008b3e:	1141                	addi	sp,sp,-16
1c008b40:	c606                	sw	ra,12(sp)
1c008b42:	c422                	sw	s0,8(sp)
1c008b44:	97ae                	add	a5,a5,a1
1c008b46:	0fe00693          	li	a3,254
1c008b4a:	02f6d663          	ble	a5,a3,1c008b76 <scalbnf+0x76>
1c008b4e:	1c0017b7          	lui	a5,0x1c001
1c008b52:	c447a403          	lw	s0,-956(a5) # 1c000c44 <PIo2+0x2e8>
1c008b56:	85aa                	mv	a1,a0
1c008b58:	8522                	mv	a0,s0
1c008b5a:	209d                	jal	1c008bc0 <copysignf>
1c008b5c:	10857553          	fmul.s	a0,a0,s0
1c008b60:	a00d                	j	1c008b82 <scalbnf+0x82>
1c008b62:	1c0017b7          	lui	a5,0x1c001
1c008b66:	c4c7a783          	lw	a5,-948(a5) # 1c000c4c <PIo2+0x2f0>
1c008b6a:	10f57553          	fmul.s	a0,a0,a5
1c008b6e:	8082                	ret
1c008b70:	872a                	mv	a4,a0
1c008b72:	83dd                	srli	a5,a5,0x17
1c008b74:	b7e9                	j	1c008b3e <scalbnf+0x3e>
1c008b76:	00f05a63          	blez	a5,1c008b8a <scalbnf+0x8a>
1c008b7a:	cf773533          	p.bclr	a0,a4,7,23
1c008b7e:	07de                	slli	a5,a5,0x17
1c008b80:	8d5d                	or	a0,a0,a5
1c008b82:	40b2                	lw	ra,12(sp)
1c008b84:	4422                	lw	s0,8(sp)
1c008b86:	0141                	addi	sp,sp,16
1c008b88:	8082                	ret
1c008b8a:	56a9                	li	a3,-22
1c008b8c:	00d7dc63          	ble	a3,a5,1c008ba4 <scalbnf+0xa4>
1c008b90:	67b1                	lui	a5,0xc
1c008b92:	35078793          	addi	a5,a5,848 # c350 <_l1_preload_size+0x5340>
1c008b96:	fab7cce3          	blt	a5,a1,1c008b4e <scalbnf+0x4e>
1c008b9a:	1c0017b7          	lui	a5,0x1c001
1c008b9e:	c4c7a403          	lw	s0,-948(a5) # 1c000c4c <PIo2+0x2f0>
1c008ba2:	bf55                	j	1c008b56 <scalbnf+0x56>
1c008ba4:	01978513          	addi	a0,a5,25
1c008ba8:	1c0017b7          	lui	a5,0x1c001
1c008bac:	c507a783          	lw	a5,-944(a5) # 1c000c50 <PIo2+0x2f4>
1c008bb0:	055e                	slli	a0,a0,0x17
1c008bb2:	cf773733          	p.bclr	a4,a4,7,23
1c008bb6:	8d59                	or	a0,a0,a4
1c008bb8:	10f57553          	fmul.s	a0,a0,a5
1c008bbc:	b7d9                	j	1c008b82 <scalbnf+0x82>
1c008bbe:	8082                	ret

1c008bc0 <copysignf>:
1c008bc0:	fc0525b3          	p.insert	a1,a0,30,0
1c008bc4:	852e                	mv	a0,a1
1c008bc6:	8082                	ret

1c008bc8 <_entry>:
1c008bc8:	7a101073          	csrw	pcmr,zero
1c008bcc:	f1402573          	csrr	a0,mhartid
1c008bd0:	01f57593          	andi	a1,a0,31
1c008bd4:	8115                	srli	a0,a0,0x5
1c008bd6:	467d                	li	a2,31
1c008bd8:	00c50463          	beq	a0,a2,1c008be0 <_entry+0x18>
1c008bdc:	4240706f          	j	1c010000 <__cluster_text_start>
1c008be0:	ffff9297          	auipc	t0,0xffff9
1c008be4:	a7028293          	addi	t0,t0,-1424 # 1c001650 <_edata>
1c008be8:	ffff9317          	auipc	t1,0xffff9
1c008bec:	c1030313          	addi	t1,t1,-1008 # 1c0017f8 <__l2_priv0_end>
1c008bf0:	0002a023          	sw	zero,0(t0)
1c008bf4:	0291                	addi	t0,t0,4
1c008bf6:	fe62ede3          	bltu	t0,t1,1c008bf0 <_entry+0x28>
1c008bfa:	ffff9117          	auipc	sp,0xffff9
1c008bfe:	86610113          	addi	sp,sp,-1946 # 1c001460 <stack>
1c008c02:	3fb010ef          	jal	ra,1c00a7fc <__rt_init>
1c008c06:	00000513          	li	a0,0
1c008c0a:	00000593          	li	a1,0
1c008c0e:	00001397          	auipc	t2,0x1
1c008c12:	99038393          	addi	t2,t2,-1648 # 1c00959e <main>
1c008c16:	000380e7          	jalr	t2
1c008c1a:	842a                	mv	s0,a0
1c008c1c:	589010ef          	jal	ra,1c00a9a4 <__rt_deinit>
1c008c20:	8522                	mv	a0,s0
1c008c22:	48d020ef          	jal	ra,1c00b8ae <exit>

1c008c26 <_fini>:
1c008c26:	8082                	ret

1c008c28 <__rt_event_enqueue>:
1c008c28:	0035f513          	andi	a0,a1,3
1c008c2c:	02051c63          	bnez	a0,1c008c64 <__rt_handle_special_event>
1c008c30:	e3ff7517          	auipc	a0,0xe3ff7
1c008c34:	3dc50513          	addi	a0,a0,988 # c <__rt_sched>
1c008c38:	0005ac23          	sw	zero,24(a1)
1c008c3c:	4110                	lw	a2,0(a0)
1c008c3e:	c601                	beqz	a2,1c008c46 <__rt_no_first>
1c008c40:	4150                	lw	a2,4(a0)
1c008c42:	ce0c                	sw	a1,24(a2)
1c008c44:	a011                	j	1c008c48 <__rt_common>

1c008c46 <__rt_no_first>:
1c008c46:	c10c                	sw	a1,0(a0)

1c008c48 <__rt_common>:
1c008c48:	c14c                	sw	a1,4(a0)
1c008c4a:	4550                	lw	a2,12(a0)
1c008c4c:	00052623          	sw	zero,12(a0)
1c008c50:	ca01                	beqz	a2,1c008c60 <enqueue_end>
1c008c52:	e3ff7517          	auipc	a0,0xe3ff7
1c008c56:	3f650513          	addi	a0,a0,1014 # 48 <__rt_thread_current>
1c008c5a:	4108                	lw	a0,0(a0)
1c008c5c:	00c51363          	bne	a0,a2,1c008c62 <thread_enqueue>

1c008c60 <enqueue_end>:
1c008c60:	8482                	jr	s1

1c008c62 <thread_enqueue>:
1c008c62:	8482                	jr	s1

1c008c64 <__rt_handle_special_event>:
1c008c64:	5571                	li	a0,-4
1c008c66:	8de9                	and	a1,a1,a0
1c008c68:	4190                	lw	a2,0(a1)
1c008c6a:	41c8                	lw	a0,4(a1)
1c008c6c:	a0d9                	j	1c008d32 <__rt_call_external_c_function>

1c008c6e <__rt_bridge_enqueue_event>:
1c008c6e:	fe812e23          	sw	s0,-4(sp)
1c008c72:	fe912c23          	sw	s1,-8(sp)
1c008c76:	fea12a23          	sw	a0,-12(sp)
1c008c7a:	feb12823          	sw	a1,-16(sp)
1c008c7e:	fec12623          	sw	a2,-20(sp)
1c008c82:	00002617          	auipc	a2,0x2
1c008c86:	0de60613          	addi	a2,a2,222 # 1c00ad60 <__rt_bridge_handle_notif>
1c008c8a:	0a8004ef          	jal	s1,1c008d32 <__rt_call_external_c_function>
1c008c8e:	ffc12403          	lw	s0,-4(sp)
1c008c92:	ff812483          	lw	s1,-8(sp)
1c008c96:	ff412503          	lw	a0,-12(sp)
1c008c9a:	ff012583          	lw	a1,-16(sp)
1c008c9e:	fec12603          	lw	a2,-20(sp)
1c008ca2:	30200073          	mret

1c008ca6 <__rt_remote_enqueue_event>:
1c008ca6:	fe812e23          	sw	s0,-4(sp)
1c008caa:	fe912c23          	sw	s1,-8(sp)
1c008cae:	fea12a23          	sw	a0,-12(sp)
1c008cb2:	feb12823          	sw	a1,-16(sp)
1c008cb6:	fec12623          	sw	a2,-20(sp)
1c008cba:	4405                	li	s0,1
1c008cbc:	ffff9497          	auipc	s1,0xffff9
1c008cc0:	af848493          	addi	s1,s1,-1288 # 1c0017b4 <__rt_fc_cluster_data>

1c008cc4 <__rt_remote_enqueue_event_loop_cluster>:
1c008cc4:	40cc                	lw	a1,4(s1)
1c008cc6:	02058d63          	beqz	a1,1c008d00 <__rt_remote_enqueue_event_loop_cluster_continue>
1c008cca:	48cc                	lw	a1,20(s1)
1c008ccc:	41c8                	lw	a0,4(a1)
1c008cce:	00050e63          	beqz	a0,1c008cea <__rt_cluster_pool_update_end>

1c008cd2 <__rt_cluster_pool_update_loop>:
1c008cd2:	5150                	lw	a2,36(a0)
1c008cd4:	e219                	bnez	a2,1c008cda <__rt_cluster_pool_update_loop_end>
1c008cd6:	5108                	lw	a0,32(a0)
1c008cd8:	fd6d                	bnez	a0,1c008cd2 <__rt_cluster_pool_update_loop>

1c008cda <__rt_cluster_pool_update_loop_end>:
1c008cda:	c501                	beqz	a0,1c008ce2 <__rt_cluster_pool_update_no_current>
1c008cdc:	5108                	lw	a0,32(a0)
1c008cde:	c1c8                	sw	a0,4(a1)
1c008ce0:	a029                	j	1c008cea <__rt_cluster_pool_update_end>

1c008ce2 <__rt_cluster_pool_update_no_current>:
1c008ce2:	0005a223          	sw	zero,4(a1)
1c008ce6:	0005a423          	sw	zero,8(a1)

1c008cea <__rt_cluster_pool_update_end>:
1c008cea:	40cc                	lw	a1,4(s1)
1c008cec:	4890                	lw	a2,16(s1)
1c008cee:	0004a223          	sw	zero,4(s1)
1c008cf2:	00062023          	sw	zero,0(a2)
1c008cf6:	00000497          	auipc	s1,0x0
1c008cfa:	00a48493          	addi	s1,s1,10 # 1c008d00 <__rt_remote_enqueue_event_loop_cluster_continue>
1c008cfe:	b72d                	j	1c008c28 <__rt_event_enqueue>

1c008d00 <__rt_remote_enqueue_event_loop_cluster_continue>:
1c008d00:	147d                	addi	s0,s0,-1
1c008d02:	00804e63          	bgtz	s0,1c008d1e <__rt_remote_enqueue_event_loop_next_cluster>
1c008d06:	ffc12403          	lw	s0,-4(sp)
1c008d0a:	ff812483          	lw	s1,-8(sp)
1c008d0e:	ff412503          	lw	a0,-12(sp)
1c008d12:	ff012583          	lw	a1,-16(sp)
1c008d16:	fec12603          	lw	a2,-20(sp)
1c008d1a:	30200073          	mret

1c008d1e <__rt_remote_enqueue_event_loop_next_cluster>:
1c008d1e:	ffff9497          	auipc	s1,0xffff9
1c008d22:	a9648493          	addi	s1,s1,-1386 # 1c0017b4 <__rt_fc_cluster_data>
1c008d26:	02800593          	li	a1,40
1c008d2a:	02b405b3          	mul	a1,s0,a1
1c008d2e:	94ae                	add	s1,s1,a1
1c008d30:	bf51                	j	1c008cc4 <__rt_remote_enqueue_event_loop_cluster>

1c008d32 <__rt_call_external_c_function>:
1c008d32:	7119                	addi	sp,sp,-128
1c008d34:	c006                	sw	ra,0(sp)
1c008d36:	c20e                	sw	gp,4(sp)
1c008d38:	c412                	sw	tp,8(sp)
1c008d3a:	c616                	sw	t0,12(sp)
1c008d3c:	c81a                	sw	t1,16(sp)
1c008d3e:	ca1e                	sw	t2,20(sp)
1c008d40:	d236                	sw	a3,36(sp)
1c008d42:	d43a                	sw	a4,40(sp)
1c008d44:	d63e                	sw	a5,44(sp)
1c008d46:	d842                	sw	a6,48(sp)
1c008d48:	da46                	sw	a7,52(sp)
1c008d4a:	dc72                	sw	t3,56(sp)
1c008d4c:	de76                	sw	t4,60(sp)
1c008d4e:	c0fa                	sw	t5,64(sp)
1c008d50:	c6fe                	sw	t6,76(sp)
1c008d52:	000600e7          	jalr	a2
1c008d56:	4082                	lw	ra,0(sp)
1c008d58:	4192                	lw	gp,4(sp)
1c008d5a:	4222                	lw	tp,8(sp)
1c008d5c:	42b2                	lw	t0,12(sp)
1c008d5e:	4342                	lw	t1,16(sp)
1c008d60:	43d2                	lw	t2,20(sp)
1c008d62:	5692                	lw	a3,36(sp)
1c008d64:	5722                	lw	a4,40(sp)
1c008d66:	57b2                	lw	a5,44(sp)
1c008d68:	5842                	lw	a6,48(sp)
1c008d6a:	58d2                	lw	a7,52(sp)
1c008d6c:	5e62                	lw	t3,56(sp)
1c008d6e:	5ef2                	lw	t4,60(sp)
1c008d70:	4f06                	lw	t5,64(sp)
1c008d72:	4fb6                	lw	t6,76(sp)
1c008d74:	6109                	addi	sp,sp,128
1c008d76:	8482                	jr	s1

1c008d78 <__rt_illegal_instr>:
1c008d78:	fe112e23          	sw	ra,-4(sp)
1c008d7c:	fea12c23          	sw	a0,-8(sp)
1c008d80:	00002517          	auipc	a0,0x2
1c008d84:	cac50513          	addi	a0,a0,-852 # 1c00aa2c <__rt_handle_illegal_instr>
1c008d88:	010000ef          	jal	ra,1c008d98 <__rt_call_c_function>
1c008d8c:	ffc12083          	lw	ra,-4(sp)
1c008d90:	ff812503          	lw	a0,-8(sp)
1c008d94:	30200073          	mret

1c008d98 <__rt_call_c_function>:
1c008d98:	7119                	addi	sp,sp,-128
1c008d9a:	c006                	sw	ra,0(sp)
1c008d9c:	c20e                	sw	gp,4(sp)
1c008d9e:	c412                	sw	tp,8(sp)
1c008da0:	c616                	sw	t0,12(sp)
1c008da2:	c81a                	sw	t1,16(sp)
1c008da4:	ca1e                	sw	t2,20(sp)
1c008da6:	ce2e                	sw	a1,28(sp)
1c008da8:	d032                	sw	a2,32(sp)
1c008daa:	d236                	sw	a3,36(sp)
1c008dac:	d43a                	sw	a4,40(sp)
1c008dae:	d63e                	sw	a5,44(sp)
1c008db0:	d842                	sw	a6,48(sp)
1c008db2:	da46                	sw	a7,52(sp)
1c008db4:	dc72                	sw	t3,56(sp)
1c008db6:	de76                	sw	t4,60(sp)
1c008db8:	c0fa                	sw	t5,64(sp)
1c008dba:	c6fe                	sw	t6,76(sp)
1c008dbc:	000500e7          	jalr	a0
1c008dc0:	4082                	lw	ra,0(sp)
1c008dc2:	4192                	lw	gp,4(sp)
1c008dc4:	4222                	lw	tp,8(sp)
1c008dc6:	42b2                	lw	t0,12(sp)
1c008dc8:	4342                	lw	t1,16(sp)
1c008dca:	43d2                	lw	t2,20(sp)
1c008dcc:	45f2                	lw	a1,28(sp)
1c008dce:	5602                	lw	a2,32(sp)
1c008dd0:	5692                	lw	a3,36(sp)
1c008dd2:	5722                	lw	a4,40(sp)
1c008dd4:	57b2                	lw	a5,44(sp)
1c008dd6:	5842                	lw	a6,48(sp)
1c008dd8:	58d2                	lw	a7,52(sp)
1c008dda:	5e62                	lw	t3,56(sp)
1c008ddc:	5ef2                	lw	t4,60(sp)
1c008dde:	4f06                	lw	t5,64(sp)
1c008de0:	4fb6                	lw	t6,76(sp)
1c008de2:	6109                	addi	sp,sp,128
1c008de4:	8082                	ret

1c008de6 <__rt_thread_start>:
1c008de6:	8526                	mv	a0,s1
1c008de8:	80ca                	mv	ra,s2
1c008dea:	8402                	jr	s0

1c008dec <__rt_thread_switch>:
1c008dec:	00152023          	sw	ra,0(a0)
1c008df0:	c140                	sw	s0,4(a0)
1c008df2:	c504                	sw	s1,8(a0)
1c008df4:	01252623          	sw	s2,12(a0)
1c008df8:	01352823          	sw	s3,16(a0)
1c008dfc:	01452a23          	sw	s4,20(a0)
1c008e00:	01552c23          	sw	s5,24(a0)
1c008e04:	01652e23          	sw	s6,28(a0)
1c008e08:	03752023          	sw	s7,32(a0)
1c008e0c:	03852223          	sw	s8,36(a0)
1c008e10:	03952423          	sw	s9,40(a0)
1c008e14:	03a52623          	sw	s10,44(a0)
1c008e18:	03b52823          	sw	s11,48(a0)
1c008e1c:	02252a23          	sw	sp,52(a0)
1c008e20:	0005a083          	lw	ra,0(a1)
1c008e24:	41c0                	lw	s0,4(a1)
1c008e26:	4584                	lw	s1,8(a1)
1c008e28:	00c5a903          	lw	s2,12(a1)
1c008e2c:	0105a983          	lw	s3,16(a1)
1c008e30:	0145aa03          	lw	s4,20(a1)
1c008e34:	0185aa83          	lw	s5,24(a1)
1c008e38:	01c5ab03          	lw	s6,28(a1)
1c008e3c:	0205ab83          	lw	s7,32(a1)
1c008e40:	0245ac03          	lw	s8,36(a1)
1c008e44:	0285ac83          	lw	s9,40(a1)
1c008e48:	02c5ad03          	lw	s10,44(a1)
1c008e4c:	0305ad83          	lw	s11,48(a1)
1c008e50:	0345a103          	lw	sp,52(a1)
1c008e54:	8082                	ret

1c008e56 <__rt_fc_socevents_handler>:
1c008e56:	7119                	addi	sp,sp,-128
1c008e58:	c022                	sw	s0,0(sp)
1c008e5a:	c226                	sw	s1,4(sp)
1c008e5c:	c42a                	sw	a0,8(sp)
1c008e5e:	c62e                	sw	a1,12(sp)
1c008e60:	c832                	sw	a2,16(sp)
1c008e62:	1a109437          	lui	s0,0x1a109
1c008e66:	5048                	lw	a0,36(s0)
1c008e68:	04000493          	li	s1,64
1c008e6c:	00955963          	ble	s1,a0,1c008e7e <__rt_fc_socevents_no_udma>
1c008e70:	ffc57613          	andi	a2,a0,-4
1c008e74:	46c62583          	lw	a1,1132(a2)
1c008e78:	4ac62403          	lw	s0,1196(a2)
1c008e7c:	8582                	jr	a1

1c008e7e <__rt_fc_socevents_no_udma>:
1c008e7e:	08d00493          	li	s1,141
1c008e82:	02a48463          	beq	s1,a0,1c008eaa <rtc_event_handler>

1c008e86 <__rt_fc_socevents_register>:
1c008e86:	00555593          	srli	a1,a0,0x5
1c008e8a:	058a                	slli	a1,a1,0x2
1c008e8c:	44c5a603          	lw	a2,1100(a1)
1c008e90:	897d                	andi	a0,a0,31
1c008e92:	80a64633          	p.bsetr	a2,a2,a0
1c008e96:	44c5a623          	sw	a2,1100(a1)

1c008e9a <__rt_fc_socevents_handler_exit>:
1c008e9a:	4402                	lw	s0,0(sp)
1c008e9c:	4492                	lw	s1,4(sp)
1c008e9e:	4522                	lw	a0,8(sp)
1c008ea0:	45b2                	lw	a1,12(sp)
1c008ea2:	4642                	lw	a2,16(sp)
1c008ea4:	6109                	addi	sp,sp,128
1c008ea6:	30200073          	mret

1c008eaa <rtc_event_handler>:
1c008eaa:	e3ff7597          	auipc	a1,0xe3ff7
1c008eae:	67a5a583          	lw	a1,1658(a1) # 524 <__rtc_handler>
1c008eb2:	fe0584e3          	beqz	a1,1c008e9a <__rt_fc_socevents_handler_exit>
1c008eb6:	00000497          	auipc	s1,0x0
1c008eba:	fe448493          	addi	s1,s1,-28 # 1c008e9a <__rt_fc_socevents_handler_exit>
1c008ebe:	d6bff06f          	j	1c008c28 <__rt_event_enqueue>

1c008ec2 <udma_event_handler>:
1c008ec2:	00157413          	andi	s0,a0,1
1c008ec6:	00155613          	srli	a2,a0,0x1
1c008eca:	8e41                	or	a2,a2,s0
1c008ecc:	e3ff7417          	auipc	s0,0xe3ff7
1c008ed0:	18040413          	addi	s0,s0,384 # 4c <periph_channels>
1c008ed4:	00561493          	slli	s1,a2,0x5
1c008ed8:	94a2                	add	s1,s1,s0
1c008eda:	4080                	lw	s0,0(s1)
1c008edc:	448c                	lw	a1,8(s1)
1c008ede:	08040f63          	beqz	s0,1c008f7c <__rt_udma_no_copy>
1c008ee2:	4c50                	lw	a2,28(s0)
1c008ee4:	4848                	lw	a0,20(s0)
1c008ee6:	04061f63          	bnez	a2,1c008f44 <dmaCmd>
1c008eea:	c088                	sw	a0,0(s1)
1c008eec:	4448                	lw	a0,12(s0)
1c008eee:	e15d                	bnez	a0,1c008f94 <handle_special_end>

1c008ef0 <resume_after_special_end>:
1c008ef0:	02058f63          	beqz	a1,1c008f2e <checkTask>
1c008ef4:	4990                	lw	a2,16(a1)
1c008ef6:	49c8                	lw	a0,20(a1)
1c008ef8:	c611                	beqz	a2,1c008f04 <__rt_udma_call_enqueue_callback_resume>
1c008efa:	00000497          	auipc	s1,0x0
1c008efe:	00a48493          	addi	s1,s1,10 # 1c008f04 <__rt_udma_call_enqueue_callback_resume>
1c008f02:	8602                	jr	a2

1c008f04 <__rt_udma_call_enqueue_callback_resume>:
1c008f04:	44d0                	lw	a2,12(s1)
1c008f06:	c488                	sw	a0,8(s1)
1c008f08:	4188                	lw	a0,0(a1)
1c008f0a:	41c4                	lw	s1,4(a1)
1c008f0c:	c208                	sw	a0,0(a2)
1c008f0e:	c244                	sw	s1,4(a2)
1c008f10:	45c4                	lw	s1,12(a1)
1c008f12:	88bd                	andi	s1,s1,15
1c008f14:	4515                	li	a0,5
1c008f16:	00a4ca63          	blt	s1,a0,1c008f2a <transfer_resume>
1c008f1a:	0064a463          	p.beqimm	s1,6,1c008f22 <dual>
1c008f1e:	0074a263          	p.beqimm	s1,7,1c008f22 <dual>

1c008f22 <dual>:
1c008f22:	51c8                	lw	a0,36(a1)
1c008f24:	c04634b3          	p.bclr	s1,a2,0,4
1c008f28:	d088                	sw	a0,32(s1)

1c008f2a <transfer_resume>:
1c008f2a:	4588                	lw	a0,8(a1)
1c008f2c:	c608                	sw	a0,8(a2)

1c008f2e <checkTask>:
1c008f2e:	4c0c                	lw	a1,24(s0)
1c008f30:	00000497          	auipc	s1,0x0
1c008f34:	f6a48493          	addi	s1,s1,-150 # 1c008e9a <__rt_fc_socevents_handler_exit>
1c008f38:	00058463          	beqz	a1,1c008f40 <checkTask+0x12>
1c008f3c:	cedff06f          	j	1c008c28 <__rt_event_enqueue>
1c008f40:	f5bff06f          	j	1c008e9a <__rt_fc_socevents_handler_exit>

1c008f44 <dmaCmd>:
1c008f44:	44cc                	lw	a1,12(s1)
1c008f46:	5048                	lw	a0,36(s0)
1c008f48:	c045b4b3          	p.bclr	s1,a1,0,4
1c008f4c:	9532                	add	a0,a0,a2
1c008f4e:	d088                	sw	a0,32(s1)
1c008f50:	d048                	sw	a0,36(s0)
1c008f52:	4008                	lw	a0,0(s0)
1c008f54:	5004                	lw	s1,32(s0)
1c008f56:	9532                	add	a0,a0,a2
1c008f58:	8c91                	sub	s1,s1,a2
1c008f5a:	00964963          	blt	a2,s1,1c008f6c <not_last>
1c008f5e:	8626                	mv	a2,s1
1c008f60:	00042e23          	sw	zero,28(s0)
1c008f64:	00061463          	bnez	a2,1c008f6c <not_last>
1c008f68:	f33ff06f          	j	1c008e9a <__rt_fc_socevents_handler_exit>

1c008f6c <not_last>:
1c008f6c:	c008                	sw	a0,0(s0)
1c008f6e:	d004                	sw	s1,32(s0)
1c008f70:	c188                	sw	a0,0(a1)
1c008f72:	c1d0                	sw	a2,4(a1)
1c008f74:	4541                	li	a0,16
1c008f76:	c588                	sw	a0,8(a1)
1c008f78:	f23ff06f          	j	1c008e9a <__rt_fc_socevents_handler_exit>

1c008f7c <__rt_udma_no_copy>:
1c008f7c:	00555593          	srli	a1,a0,0x5
1c008f80:	058a                	slli	a1,a1,0x2
1c008f82:	44c5a603          	lw	a2,1100(a1)
1c008f86:	897d                	andi	a0,a0,31
1c008f88:	80a64633          	p.bsetr	a2,a2,a0
1c008f8c:	44c5a623          	sw	a2,1100(a1)
1c008f90:	f0bff06f          	j	1c008e9a <__rt_fc_socevents_handler_exit>

1c008f94 <handle_special_end>:
1c008f94:	00352563          	p.beqimm	a0,3,1c008f9e <i2c_step1>
1c008f98:	02452163          	p.beqimm	a0,4,1c008fba <i2c_step2>
1c008f9c:	bf91                	j	1c008ef0 <resume_after_special_end>

1c008f9e <i2c_step1>:
1c008f9e:	5408                	lw	a0,40(s0)
1c008fa0:	c448                	sw	a0,12(s0)
1c008fa2:	4088                	lw	a0,0(s1)
1c008fa4:	c848                	sw	a0,20(s0)
1c008fa6:	c080                	sw	s0,0(s1)
1c008fa8:	44d0                	lw	a2,12(s1)
1c008faa:	4008                	lw	a0,0(s0)
1c008fac:	c208                	sw	a0,0(a2)
1c008fae:	5048                	lw	a0,36(s0)
1c008fb0:	c248                	sw	a0,4(a2)
1c008fb2:	4408                	lw	a0,8(s0)
1c008fb4:	c608                	sw	a0,8(a2)
1c008fb6:	ee5ff06f          	j	1c008e9a <__rt_fc_socevents_handler_exit>

1c008fba <i2c_step2>:
1c008fba:	00042623          	sw	zero,12(s0)
1c008fbe:	4088                	lw	a0,0(s1)
1c008fc0:	c848                	sw	a0,20(s0)
1c008fc2:	c080                	sw	s0,0(s1)
1c008fc4:	02000613          	li	a2,32
1c008fc8:	c070                	sw	a2,68(s0)
1c008fca:	44d0                	lw	a2,12(s1)
1c008fcc:	04440513          	addi	a0,s0,68
1c008fd0:	c208                	sw	a0,0(a2)
1c008fd2:	4505                	li	a0,1
1c008fd4:	c248                	sw	a0,4(a2)
1c008fd6:	4541                	li	a0,16
1c008fd8:	c608                	sw	a0,8(a2)
1c008fda:	ec1ff06f          	j	1c008e9a <__rt_fc_socevents_handler_exit>

1c008fde <bit_rev_radix2>:
1c008fde:	87aa                	mv	a5,a0
1c008fe0:	0027f713          	andi	a4,a5,2
1c008fe4:	fc153533          	p.bclr	a0,a0,30,1
1c008fe8:	052a                	slli	a0,a0,0xa
1c008fea:	c319                	beqz	a4,1c008ff0 <bit_rev_radix2+0x12>
1c008fec:	c0954533          	p.bset	a0,a0,0,9
1c008ff0:	0047f713          	andi	a4,a5,4
1c008ff4:	c319                	beqz	a4,1c008ffa <bit_rev_radix2+0x1c>
1c008ff6:	c0854533          	p.bset	a0,a0,0,8
1c008ffa:	0087f713          	andi	a4,a5,8
1c008ffe:	c319                	beqz	a4,1c009004 <bit_rev_radix2+0x26>
1c009000:	c0754533          	p.bset	a0,a0,0,7
1c009004:	0107f713          	andi	a4,a5,16
1c009008:	c319                	beqz	a4,1c00900e <bit_rev_radix2+0x30>
1c00900a:	c0654533          	p.bset	a0,a0,0,6
1c00900e:	0207f713          	andi	a4,a5,32
1c009012:	c319                	beqz	a4,1c009018 <bit_rev_radix2+0x3a>
1c009014:	c0554533          	p.bset	a0,a0,0,5
1c009018:	0407f713          	andi	a4,a5,64
1c00901c:	c319                	beqz	a4,1c009022 <bit_rev_radix2+0x44>
1c00901e:	c0454533          	p.bset	a0,a0,0,4
1c009022:	0807f713          	andi	a4,a5,128
1c009026:	c319                	beqz	a4,1c00902c <bit_rev_radix2+0x4e>
1c009028:	c0354533          	p.bset	a0,a0,0,3
1c00902c:	1007f713          	andi	a4,a5,256
1c009030:	c319                	beqz	a4,1c009036 <bit_rev_radix2+0x58>
1c009032:	c0254533          	p.bset	a0,a0,0,2
1c009036:	2007f713          	andi	a4,a5,512
1c00903a:	c319                	beqz	a4,1c009040 <bit_rev_radix2+0x62>
1c00903c:	c0154533          	p.bset	a0,a0,0,1
1c009040:	4007f793          	andi	a5,a5,1024
1c009044:	c399                	beqz	a5,1c00904a <bit_rev_radix2+0x6c>
1c009046:	c0054533          	p.bset	a0,a0,0,0
1c00904a:	8082                	ret

1c00904c <fft_radix2>:
1c00904c:	7139                	addi	sp,sp,-64
1c00904e:	100017b7          	lui	a5,0x10001
1c009052:	de22                	sw	s0,60(sp)
1c009054:	6e89                	lui	t4,0x2
1c009056:	842a                	mv	s0,a0
1c009058:	dc26                	sw	s1,56(sp)
1c00905a:	cc66                	sw	s9,24(sp)
1c00905c:	da4a                	sw	s2,52(sp)
1c00905e:	d84e                	sw	s3,48(sp)
1c009060:	d652                	sw	s4,44(sp)
1c009062:	d456                	sw	s5,40(sp)
1c009064:	d25a                	sw	s6,36(sp)
1c009066:	d05e                	sw	s7,32(sp)
1c009068:	ce62                	sw	s8,28(sp)
1c00906a:	ca6a                	sw	s10,20(sp)
1c00906c:	c86e                	sw	s11,16(sp)
1c00906e:	84ae                	mv	s1,a1
1c009070:	01878c93          	addi	s9,a5,24 # 10001018 <twiddle_factors>
1c009074:	01878513          	addi	a0,a5,24
1c009078:	8822                	mv	a6,s0
1c00907a:	004e8f13          	addi	t5,t4,4 # 2004 <__rt_hyper_pending_tasks_last+0x1a9c>
1c00907e:	400bd0fb          	lp.setupi	x1,1024,1c0090ac <fft_radix2+0x60>
1c009082:	87c2                	mv	a5,a6
1c009084:	21d7f70b          	p.lw	a4,t4(a5!)
1c009088:	410c                	lw	a1,0(a0)
1c00908a:	4150                	lw	a2,4(a0)
1c00908c:	0007a303          	lw	t1,0(a5)
1c009090:	86c2                	mv	a3,a6
1c009092:	0521                	addi	a0,a0,8
1c009094:	086778d3          	fsub.s	a7,a4,t1
1c009098:	00677753          	fadd.s	a4,a4,t1
1c00909c:	0821                	addi	a6,a6,8
1c00909e:	1115f5d3          	fmul.s	a1,a1,a7
1c0090a2:	11167653          	fmul.s	a2,a2,a7
1c0090a6:	00e6ef2b          	p.sw	a4,t5(a3!)
1c0090aa:	c38c                	sw	a1,0(a5)
1c0090ac:	c290                	sw	a2,0(a3)
1c0090ae:	10001c37          	lui	s8,0x10001
1c0090b2:	4ba5                	li	s7,9
1c0090b4:	4d09                	li	s10,2
1c0090b6:	20000a13          	li	s4,512
1c0090ba:	01cc0c13          	addi	s8,s8,28 # 1000101c <twiddle_factors+0x4>
1c0090be:	c626                	sw	s1,12(sp)
1c0090c0:	09a05163          	blez	s10,1c009142 <fft_radix2+0xf6>
1c0090c4:	003a1993          	slli	s3,s4,0x3
1c0090c8:	004a1b13          	slli	s6,s4,0x4
1c0090cc:	84a2                	mv	s1,s0
1c0090ce:	003d1d93          	slli	s11,s10,0x3
1c0090d2:	00498a93          	addi	s5,s3,4
1c0090d6:	896a                	mv	s2,s10
1c0090d8:	0349407b          	lp.setup	x0,s2,1c009140 <fft_radix2+0xf4>
1c0090dc:	4785                	li	a5,1
1c0090de:	00998eb3          	add	t4,s3,s1
1c0090e2:	00448e13          	addi	t3,s1,4
1c0090e6:	009a8333          	add	t1,s5,s1
1c0090ea:	8fe2                	mv	t6,s8
1c0090ec:	8f66                	mv	t5,s9
1c0090ee:	88a6                	mv	a7,s1
1c0090f0:	04fa6833          	p.max	a6,s4,a5
1c0090f4:	024840fb          	lp.setup	x1,a6,1c00913c <fft_radix2+0xf0>
1c0090f8:	000e2383          	lw	t2,0(t3)
1c0090fc:	00032603          	lw	a2,0(t1)
1c009100:	0008a283          	lw	t0,0(a7)
1c009104:	000ea583          	lw	a1,0(t4)
1c009108:	08c3f7d3          	fsub.s	a5,t2,a2
1c00910c:	21bff68b          	p.lw	a3,s11(t6!)
1c009110:	08b2f753          	fsub.s	a4,t0,a1
1c009114:	00b2f5d3          	fadd.s	a1,t0,a1
1c009118:	21bf750b          	p.lw	a0,s11(t5!)
1c00911c:	10d7f2d3          	fmul.s	t0,a5,a3
1c009120:	00c3f653          	fadd.s	a2,t2,a2
1c009124:	10d776d3          	fmul.s	a3,a4,a3
1c009128:	00b8a42b          	p.sw	a1,8(a7!)
1c00912c:	28a77747          	fmsub.s	a4,a4,a0,t0
1c009130:	00ce242b          	p.sw	a2,8(t3!)
1c009134:	68a7f7c3          	fmadd.s	a5,a5,a0,a3
1c009138:	00eea42b          	p.sw	a4,8(t4!)
1c00913c:	00f3242b          	p.sw	a5,8(t1!)
1c009140:	94da                	add	s1,s1,s6
1c009142:	1bfd                	addi	s7,s7,-1
1c009144:	401a5a13          	srai	s4,s4,0x1
1c009148:	0d06                	slli	s10,s10,0x1
1c00914a:	f60b9be3          	bnez	s7,1c0090c0 <fft_radix2+0x74>
1c00914e:	44b2                	lw	s1,12(sp)
1c009150:	8526                	mv	a0,s1
1c009152:	85a6                	mv	a1,s1
1c009154:	400a50fb          	lp.setupi	x1,1024,1c00917c <fft_radix2+0x130>
1c009158:	4410                	lw	a2,8(s0)
1c00915a:	4018                	lw	a4,0(s0)
1c00915c:	405c                	lw	a5,4(s0)
1c00915e:	4454                	lw	a3,12(s0)
1c009160:	00c778d3          	fadd.s	a7,a4,a2
1c009164:	08c77753          	fsub.s	a4,a4,a2
1c009168:	00d7f653          	fadd.s	a2,a5,a3
1c00916c:	08d7f7d3          	fsub.s	a5,a5,a3
1c009170:	0441                	addi	s0,s0,16
1c009172:	0115a023          	sw	a7,0(a1)
1c009176:	c598                	sw	a4,8(a1)
1c009178:	c1d0                	sw	a2,4(a1)
1c00917a:	c5dc                	sw	a5,12(a1)
1c00917c:	05c1                	addi	a1,a1,16
1c00917e:	10000637          	lui	a2,0x10000
1c009182:	6fc1                	lui	t6,0x10
1c009184:	01860613          	addi	a2,a2,24 # 10000018 <bit_rev_radix2_LUT>
1c009188:	4781                	li	a5,0
1c00918a:	1ffd                	addi	t6,t6,-1
1c00918c:	20000f13          	li	t5,512
1c009190:	069f40fb          	lp.setup	x1,t5,1c009262 <fft_radix2+0x216>
1c009194:	4214                	lw	a3,0(a2)
1c009196:	4258                	lw	a4,4(a2)
1c009198:	00178a13          	addi	s4,a5,1
1c00919c:	01f6f833          	and	a6,a3,t6
1c0091a0:	00381593          	slli	a1,a6,0x3
1c0091a4:	82c1                	srli	a3,a3,0x10
1c0091a6:	01f77433          	and	s0,a4,t6
1c0091aa:	00b482b3          	add	t0,s1,a1
1c0091ae:	00369593          	slli	a1,a3,0x3
1c0091b2:	00b48eb3          	add	t4,s1,a1
1c0091b6:	8341                	srli	a4,a4,0x10
1c0091b8:	00341593          	slli	a1,s0,0x3
1c0091bc:	00b48e33          	add	t3,s1,a1
1c0091c0:	00371593          	slli	a1,a4,0x3
1c0091c4:	00378913          	addi	s2,a5,3
1c0091c8:	00278993          	addi	s3,a5,2
1c0091cc:	95a6                	add	a1,a1,s1
1c0091ce:	0307f263          	bleu	a6,a5,1c0091f2 <fft_radix2+0x1a6>
1c0091d2:	0002a803          	lw	a6,0(t0)
1c0091d6:	0042a883          	lw	a7,4(t0)
1c0091da:	00052303          	lw	t1,0(a0)
1c0091de:	00452383          	lw	t2,4(a0)
1c0091e2:	01052023          	sw	a6,0(a0)
1c0091e6:	01152223          	sw	a7,4(a0)
1c0091ea:	0062a023          	sw	t1,0(t0)
1c0091ee:	0072a223          	sw	t2,4(t0)
1c0091f2:	02da7263          	bleu	a3,s4,1c009216 <fft_radix2+0x1ca>
1c0091f6:	000ea803          	lw	a6,0(t4)
1c0091fa:	004ea883          	lw	a7,4(t4)
1c0091fe:	00852303          	lw	t1,8(a0)
1c009202:	00c52383          	lw	t2,12(a0)
1c009206:	01052423          	sw	a6,8(a0)
1c00920a:	01152623          	sw	a7,12(a0)
1c00920e:	006ea023          	sw	t1,0(t4)
1c009212:	007ea223          	sw	t2,4(t4)
1c009216:	0289f263          	bleu	s0,s3,1c00923a <fft_radix2+0x1ee>
1c00921a:	000e2803          	lw	a6,0(t3)
1c00921e:	004e2883          	lw	a7,4(t3)
1c009222:	01052303          	lw	t1,16(a0)
1c009226:	01452383          	lw	t2,20(a0)
1c00922a:	01052823          	sw	a6,16(a0)
1c00922e:	01152a23          	sw	a7,20(a0)
1c009232:	006e2023          	sw	t1,0(t3)
1c009236:	007e2223          	sw	t2,4(t3)
1c00923a:	02e97263          	bleu	a4,s2,1c00925e <fft_radix2+0x212>
1c00923e:	0005a803          	lw	a6,0(a1)
1c009242:	0045a883          	lw	a7,4(a1)
1c009246:	01852303          	lw	t1,24(a0)
1c00924a:	01c52383          	lw	t2,28(a0)
1c00924e:	01052c23          	sw	a6,24(a0)
1c009252:	01152e23          	sw	a7,28(a0)
1c009256:	0065a023          	sw	t1,0(a1)
1c00925a:	0075a223          	sw	t2,4(a1)
1c00925e:	0791                	addi	a5,a5,4
1c009260:	0621                	addi	a2,a2,8
1c009262:	02050513          	addi	a0,a0,32
1c009266:	5472                	lw	s0,60(sp)
1c009268:	54e2                	lw	s1,56(sp)
1c00926a:	5952                	lw	s2,52(sp)
1c00926c:	59c2                	lw	s3,48(sp)
1c00926e:	5a32                	lw	s4,44(sp)
1c009270:	5aa2                	lw	s5,40(sp)
1c009272:	5b12                	lw	s6,36(sp)
1c009274:	5b82                	lw	s7,32(sp)
1c009276:	4c72                	lw	s8,28(sp)
1c009278:	4ce2                	lw	s9,24(sp)
1c00927a:	4d52                	lw	s10,20(sp)
1c00927c:	4dc2                	lw	s11,16(sp)
1c00927e:	6121                	addi	sp,sp,64
1c009280:	8082                	ret

1c009282 <compute_twiddles>:
1c009282:	1c0017b7          	lui	a5,0x1c001
1c009286:	1101                	addi	sp,sp,-32
1c009288:	9887a503          	lw	a0,-1656(a5) # 1c000988 <PIo2+0x2c>
1c00928c:	1c0017b7          	lui	a5,0x1c001
1c009290:	c84a                	sw	s2,16(sp)
1c009292:	c64e                	sw	s3,12(sp)
1c009294:	c452                	sw	s4,8(sp)
1c009296:	100019b7          	lui	s3,0x10001
1c00929a:	98c7aa03          	lw	s4,-1652(a5) # 1c00098c <PIo2+0x30>
1c00929e:	10001937          	lui	s2,0x10001
1c0092a2:	1c0017b7          	lui	a5,0x1c001
1c0092a6:	ca26                	sw	s1,20(sp)
1c0092a8:	c256                	sw	s5,4(sp)
1c0092aa:	c05a                	sw	s6,0(sp)
1c0092ac:	ce06                	sw	ra,28(sp)
1c0092ae:	cc22                	sw	s0,24(sp)
1c0092b0:	9907ab03          	lw	s6,-1648(a5) # 1c000990 <PIo2+0x34>
1c0092b4:	01898993          	addi	s3,s3,24 # 10001018 <twiddle_factors>
1c0092b8:	01c90913          	addi	s2,s2,28 # 1000101c <twiddle_factors+0x4>
1c0092bc:	4481                	li	s1,0
1c0092be:	40000a93          	li	s5,1024
1c0092c2:	a821                	j	1c0092da <compute_twiddles+0x58>
1c0092c4:	d004f453          	fcvt.s.w	s0,s1
1c0092c8:	11647453          	fmul.s	s0,s0,s6
1c0092cc:	8522                	mv	a0,s0
1c0092ce:	dd3fe0ef          	jal	ra,1c0080a0 <cosf>
1c0092d2:	8a2a                	mv	s4,a0
1c0092d4:	8522                	mv	a0,s0
1c0092d6:	e27fe0ef          	jal	ra,1c0080fc <sinf>
1c0092da:	0149a42b          	p.sw	s4,8(s3!)
1c0092de:	00a9242b          	p.sw	a0,8(s2!)
1c0092e2:	0485                	addi	s1,s1,1
1c0092e4:	ff5490e3          	bne	s1,s5,1c0092c4 <compute_twiddles+0x42>
1c0092e8:	40f2                	lw	ra,28(sp)
1c0092ea:	4462                	lw	s0,24(sp)
1c0092ec:	44d2                	lw	s1,20(sp)
1c0092ee:	4942                	lw	s2,16(sp)
1c0092f0:	49b2                	lw	s3,12(sp)
1c0092f2:	4a22                	lw	s4,8(sp)
1c0092f4:	4a92                	lw	s5,4(sp)
1c0092f6:	4b02                	lw	s6,0(sp)
1c0092f8:	6105                	addi	sp,sp,32
1c0092fa:	8082                	ret

1c0092fc <end_of_call>:
1c0092fc:	1c0017b7          	lui	a5,0x1c001
1c009300:	4705                	li	a4,1
1c009302:	72e7ac23          	sw	a4,1848(a5) # 1c001738 <done>
1c009306:	8082                	ret

1c009308 <cluster_entry>:
1c009308:	4705                	li	a4,1
1c00930a:	002047b7          	lui	a5,0x204
1c00930e:	08e7a223          	sw	a4,132(a5) # 204084 <__l1_heap_size+0x1eb09c>
1c009312:	20078693          	addi	a3,a5,512
1c009316:	c298                	sw	a4,0(a3)
1c009318:	20c78693          	addi	a3,a5,524
1c00931c:	c298                	sw	a4,0(a3)
1c00931e:	22078713          	addi	a4,a5,544
1c009322:	10100693          	li	a3,257
1c009326:	c314                	sw	a3,0(a4)
1c009328:	22c78793          	addi	a5,a5,556
1c00932c:	c394                	sw	a3,0(a5)
1c00932e:	1c0097b7          	lui	a5,0x1c009
1c009332:	34c78793          	addi	a5,a5,844 # 1c00934c <main_fn>
1c009336:	002046b7          	lui	a3,0x204
1c00933a:	08f6a023          	sw	a5,128(a3) # 204080 <__l1_heap_size+0x1eb098>
1c00933e:	002047b7          	lui	a5,0x204
1c009342:	0807a023          	sw	zero,128(a5) # 204080 <__l1_heap_size+0x1eb098>
1c009346:	01c76783          	p.elw	a5,28(a4) # 80001c <__l1_heap_size+0x7e7034>
1c00934a:	8082                	ret

1c00934c <main_fn>:
1c00934c:	7135                	addi	sp,sp,-160
1c00934e:	f14027f3          	csrr	a5,mhartid
1c009352:	cf06                	sw	ra,156(sp)
1c009354:	cd22                	sw	s0,152(sp)
1c009356:	cb26                	sw	s1,148(sp)
1c009358:	c94a                	sw	s2,144(sp)
1c00935a:	c74e                	sw	s3,140(sp)
1c00935c:	c552                	sw	s4,136(sp)
1c00935e:	c356                	sw	s5,132(sp)
1c009360:	c15a                	sw	s6,128(sp)
1c009362:	dede                	sw	s7,124(sp)
1c009364:	dce2                	sw	s8,120(sp)
1c009366:	dae6                	sw	s9,116(sp)
1c009368:	d8ea                	sw	s10,112(sp)
1c00936a:	d6ee                	sw	s11,108(sp)
1c00936c:	f457b7b3          	p.bclr	a5,a5,26,5
1c009370:	20078863          	beqz	a5,1c009580 <main_fn+0x234>
1c009374:	f14027f3          	csrr	a5,mhartid
1c009378:	477d                	li	a4,31
1c00937a:	ca5797b3          	p.extractu	a5,a5,5,5
1c00937e:	00e78863          	beq	a5,a4,1c00938e <main_fn+0x42>
1c009382:	002047b7          	lui	a5,0x204
1c009386:	20078793          	addi	a5,a5,512 # 204200 <__l1_heap_size+0x1eb218>
1c00938a:	01c7e703          	p.elw	a4,28(a5)
1c00938e:	f14027f3          	csrr	a5,mhartid
1c009392:	8795                	srai	a5,a5,0x5
1c009394:	f267b933          	p.bclr	s2,a5,25,6
1c009398:	4401                	li	s0,0
1c00939a:	4981                	li	s3,0
1c00939c:	4a01                	li	s4,0
1c00939e:	4a81                	li	s5,0
1c0093a0:	4b01                	li	s6,0
1c0093a2:	4b81                	li	s7,0
1c0093a4:	4c01                	li	s8,0
1c0093a6:	4c81                	li	s9,0
1c0093a8:	100034b7          	lui	s1,0x10003
1c0093ac:	8dca                	mv	s11,s2
1c0093ae:	8d4a                	mv	s10,s2
1c0093b0:	c24a                	sw	s2,4(sp)
1c0093b2:	c43e                	sw	a5,8(sp)
1c0093b4:	c64a                	sw	s2,12(sp)
1c0093b6:	a04d                	j	1c009458 <main_fn+0x10c>
1c0093b8:	102007b7          	lui	a5,0x10200
1c0093bc:	4705                	li	a4,1
1c0093be:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1f93e8>
1c0093c2:	02e7a023          	sw	a4,32(a5)
1c0093c6:	4781                	li	a5,0
1c0093c8:	79f79073          	csrw	pccr31,a5
1c0093cc:	47fd                	li	a5,31
1c0093ce:	0afd8c63          	beq	s11,a5,1c009486 <main_fn+0x13a>
1c0093d2:	102007b7          	lui	a5,0x10200
1c0093d6:	4705                	li	a4,1
1c0093d8:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1f93e8>
1c0093dc:	00e7ac23          	sw	a4,24(a5)
1c0093e0:	478d                	li	a5,3
1c0093e2:	cc179073          	csrw	0xcc1,a5
1c0093e6:	01848593          	addi	a1,s1,24 # 10003018 <Input_Signal>
1c0093ea:	01848513          	addi	a0,s1,24
1c0093ee:	39b9                	jal	1c00904c <fft_radix2>
1c0093f0:	47fd                	li	a5,31
1c0093f2:	0afd0a63          	beq	s10,a5,1c0094a6 <main_fn+0x15a>
1c0093f6:	102007b7          	lui	a5,0x10200
1c0093fa:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1f93e8>
1c0093fe:	0007a023          	sw	zero,0(a5)
1c009402:	4781                	li	a5,0
1c009404:	cc179073          	csrw	0xcc1,a5
1c009408:	0848                	addi	a0,sp,20
1c00940a:	346010ef          	jal	ra,1c00a750 <rt_perf_save>
1c00940e:	4785                	li	a5,1
1c009410:	0487d163          	ble	s0,a5,1c009452 <main_fn+0x106>
1c009414:	4692                	lw	a3,4(sp)
1c009416:	47fd                	li	a5,31
1c009418:	08f68f63          	beq	a3,a5,1c0094b6 <main_fn+0x16a>
1c00941c:	102007b7          	lui	a5,0x10200
1c009420:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1f93e8>
1c009424:	0087a783          	lw	a5,8(a5)
1c009428:	9cbe                	add	s9,s9,a5
1c00942a:	78102773          	csrr	a4,pccr1
1c00942e:	9c3a                	add	s8,s8,a4
1c009430:	780026f3          	csrr	a3,pccr0
1c009434:	9bb6                	add	s7,s7,a3
1c009436:	78c026f3          	csrr	a3,pccr12
1c00943a:	9b36                	add	s6,s6,a3
1c00943c:	790026f3          	csrr	a3,pccr16
1c009440:	9ab6                	add	s5,s5,a3
1c009442:	782026f3          	csrr	a3,pccr2
1c009446:	9a36                	add	s4,s4,a3
1c009448:	784027f3          	csrr	a5,pccr4
1c00944c:	99be                	add	s3,s3,a5
1c00944e:	06642963          	p.beqimm	s0,6,1c0094c0 <main_fn+0x174>
1c009452:	0405                	addi	s0,s0,1
1c009454:	0e742a63          	p.beqimm	s0,7,1c009548 <main_fn+0x1fc>
1c009458:	0848                	addi	a0,sp,20
1c00945a:	2e0010ef          	jal	ra,1c00a73a <rt_perf_init>
1c00945e:	000315b7          	lui	a1,0x31
1c009462:	05dd                	addi	a1,a1,23
1c009464:	0848                	addi	a0,sp,20
1c009466:	2e2010ef          	jal	ra,1c00a748 <rt_perf_conf>
1c00946a:	47fd                	li	a5,31
1c00946c:	f4f916e3          	bne	s2,a5,1c0093b8 <main_fn+0x6c>
1c009470:	4785                	li	a5,1
1c009472:	1a10b737          	lui	a4,0x1a10b
1c009476:	02f72023          	sw	a5,32(a4) # 1a10b020 <__l1_end+0xa104008>
1c00947a:	4781                	li	a5,0
1c00947c:	79f79073          	csrw	pccr31,a5
1c009480:	47fd                	li	a5,31
1c009482:	f4fd98e3          	bne	s11,a5,1c0093d2 <main_fn+0x86>
1c009486:	4785                	li	a5,1
1c009488:	1a10b737          	lui	a4,0x1a10b
1c00948c:	00f72c23          	sw	a5,24(a4) # 1a10b018 <__l1_end+0xa104000>
1c009490:	478d                	li	a5,3
1c009492:	cc179073          	csrw	0xcc1,a5
1c009496:	01848593          	addi	a1,s1,24
1c00949a:	01848513          	addi	a0,s1,24
1c00949e:	367d                	jal	1c00904c <fft_radix2>
1c0094a0:	47fd                	li	a5,31
1c0094a2:	f4fd1ae3          	bne	s10,a5,1c0093f6 <main_fn+0xaa>
1c0094a6:	1a10b7b7          	lui	a5,0x1a10b
1c0094aa:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa103fe8>
1c0094ae:	4781                	li	a5,0
1c0094b0:	cc179073          	csrw	0xcc1,a5
1c0094b4:	bf91                	j	1c009408 <main_fn+0xbc>
1c0094b6:	1a10b7b7          	lui	a5,0x1a10b
1c0094ba:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa103ff0>
1c0094be:	b7ad                	j	1c009428 <main_fn+0xdc>
1c0094c0:	4495                	li	s1,5
1c0094c2:	029cd633          	divu	a2,s9,s1
1c0094c6:	f1402473          	csrr	s0,mhartid
1c0094ca:	1c001537          	lui	a0,0x1c001
1c0094ce:	f4543433          	p.bclr	s0,s0,26,5
1c0094d2:	85a2                	mv	a1,s0
1c0094d4:	99450513          	addi	a0,a0,-1644 # 1c000994 <PIo2+0x38>
1c0094d8:	48e020ef          	jal	ra,1c00b966 <printf>
1c0094dc:	029c5633          	divu	a2,s8,s1
1c0094e0:	1c001537          	lui	a0,0x1c001
1c0094e4:	85a2                	mv	a1,s0
1c0094e6:	9a850513          	addi	a0,a0,-1624 # 1c0009a8 <PIo2+0x4c>
1c0094ea:	47c020ef          	jal	ra,1c00b966 <printf>
1c0094ee:	029bd633          	divu	a2,s7,s1
1c0094f2:	1c001537          	lui	a0,0x1c001
1c0094f6:	85a2                	mv	a1,s0
1c0094f8:	9bc50513          	addi	a0,a0,-1604 # 1c0009bc <PIo2+0x60>
1c0094fc:	46a020ef          	jal	ra,1c00b966 <printf>
1c009500:	029b5633          	divu	a2,s6,s1
1c009504:	1c001537          	lui	a0,0x1c001
1c009508:	85a2                	mv	a1,s0
1c00950a:	9d850513          	addi	a0,a0,-1576 # 1c0009d8 <PIo2+0x7c>
1c00950e:	458020ef          	jal	ra,1c00b966 <printf>
1c009512:	029ad633          	divu	a2,s5,s1
1c009516:	1c001537          	lui	a0,0x1c001
1c00951a:	85a2                	mv	a1,s0
1c00951c:	9f050513          	addi	a0,a0,-1552 # 1c0009f0 <PIo2+0x94>
1c009520:	446020ef          	jal	ra,1c00b966 <printf>
1c009524:	029a5633          	divu	a2,s4,s1
1c009528:	1c001537          	lui	a0,0x1c001
1c00952c:	85a2                	mv	a1,s0
1c00952e:	a0850513          	addi	a0,a0,-1528 # 1c000a08 <PIo2+0xac>
1c009532:	434020ef          	jal	ra,1c00b966 <printf>
1c009536:	0299d633          	divu	a2,s3,s1
1c00953a:	1c001537          	lui	a0,0x1c001
1c00953e:	85a2                	mv	a1,s0
1c009540:	a2050513          	addi	a0,a0,-1504 # 1c000a20 <PIo2+0xc4>
1c009544:	422020ef          	jal	ra,1c00b966 <printf>
1c009548:	f14027f3          	csrr	a5,mhartid
1c00954c:	477d                	li	a4,31
1c00954e:	ca5797b3          	p.extractu	a5,a5,5,5
1c009552:	00e78863          	beq	a5,a4,1c009562 <main_fn+0x216>
1c009556:	002047b7          	lui	a5,0x204
1c00955a:	20078793          	addi	a5,a5,512 # 204200 <__l1_heap_size+0x1eb218>
1c00955e:	01c7e703          	p.elw	a4,28(a5)
1c009562:	40fa                	lw	ra,156(sp)
1c009564:	446a                	lw	s0,152(sp)
1c009566:	44da                	lw	s1,148(sp)
1c009568:	494a                	lw	s2,144(sp)
1c00956a:	49ba                	lw	s3,140(sp)
1c00956c:	4a2a                	lw	s4,136(sp)
1c00956e:	4a9a                	lw	s5,132(sp)
1c009570:	4b0a                	lw	s6,128(sp)
1c009572:	5bf6                	lw	s7,124(sp)
1c009574:	5c66                	lw	s8,120(sp)
1c009576:	5cd6                	lw	s9,116(sp)
1c009578:	5d46                	lw	s10,112(sp)
1c00957a:	5db6                	lw	s11,108(sp)
1c00957c:	610d                	addi	sp,sp,160
1c00957e:	8082                	ret
1c009580:	100004b7          	lui	s1,0x10000
1c009584:	39fd                	jal	1c009282 <compute_twiddles>
1c009586:	01848493          	addi	s1,s1,24 # 10000018 <bit_rev_radix2_LUT>
1c00958a:	4401                	li	s0,0
1c00958c:	8522                	mv	a0,s0
1c00958e:	3c81                	jal	1c008fde <bit_rev_radix2>
1c009590:	0405                	addi	s0,s0,1
1c009592:	00a4912b          	p.sh	a0,2(s1!)
1c009596:	80040793          	addi	a5,s0,-2048
1c00959a:	fbed                	bnez	a5,1c00958c <main_fn+0x240>
1c00959c:	bbe1                	j	1c009374 <main_fn+0x28>

1c00959e <main>:
1c00959e:	1101                	addi	sp,sp,-32
1c0095a0:	4591                	li	a1,4
1c0095a2:	e3ff7517          	auipc	a0,0xe3ff7
1c0095a6:	a6a50513          	addi	a0,a0,-1430 # c <__rt_sched>
1c0095aa:	ce06                	sw	ra,28(sp)
1c0095ac:	cc22                	sw	s0,24(sp)
1c0095ae:	ca26                	sw	s1,20(sp)
1c0095b0:	c84a                	sw	s2,16(sp)
1c0095b2:	20c1                	jal	1c009672 <rt_event_alloc>
1c0095b4:	e941                	bnez	a0,1c009644 <main+0xa6>
1c0095b6:	4681                	li	a3,0
1c0095b8:	4601                	li	a2,0
1c0095ba:	4581                	li	a1,0
1c0095bc:	892a                	mv	s2,a0
1c0095be:	4505                	li	a0,1
1c0095c0:	6cf000ef          	jal	ra,1c00a48e <rt_cluster_mount>
1c0095c4:	6589                	lui	a1,0x2
1c0095c6:	40058593          	addi	a1,a1,1024 # 2400 <__rt_hyper_pending_tasks_last+0x1e98>
1c0095ca:	450d                	li	a0,3
1c0095cc:	2935                	jal	1c009a08 <rt_alloc>
1c0095ce:	842a                	mv	s0,a0
1c0095d0:	c935                	beqz	a0,1c009644 <main+0xa6>
1c0095d2:	1c0095b7          	lui	a1,0x1c009
1c0095d6:	4601                	li	a2,0
1c0095d8:	2fc58593          	addi	a1,a1,764 # 1c0092fc <end_of_call>
1c0095dc:	e3ff7517          	auipc	a0,0xe3ff7
1c0095e0:	a3050513          	addi	a0,a0,-1488 # c <__rt_sched>
1c0095e4:	2211                	jal	1c0096e8 <rt_event_get>
1c0095e6:	1c009637          	lui	a2,0x1c009
1c0095ea:	c02a                	sw	a0,0(sp)
1c0095ec:	40000793          	li	a5,1024
1c0095f0:	4881                	li	a7,0
1c0095f2:	40000813          	li	a6,1024
1c0095f6:	8722                	mv	a4,s0
1c0095f8:	4681                	li	a3,0
1c0095fa:	30860613          	addi	a2,a2,776 # 1c009308 <cluster_entry>
1c0095fe:	4581                	li	a1,0
1c009600:	4501                	li	a0,0
1c009602:	1c0014b7          	lui	s1,0x1c001
1c009606:	5f9000ef          	jal	ra,1c00a3fe <rt_cluster_call>
1c00960a:	73848493          	addi	s1,s1,1848 # 1c001738 <done>
1c00960e:	409c                	lw	a5,0(s1)
1c009610:	ef89                	bnez	a5,1c00962a <main+0x8c>
1c009612:	30047473          	csrrci	s0,mstatus,8
1c009616:	4585                	li	a1,1
1c009618:	e3ff7517          	auipc	a0,0xe3ff7
1c00961c:	9f450513          	addi	a0,a0,-1548 # c <__rt_sched>
1c009620:	2225                	jal	1c009748 <__rt_event_execute>
1c009622:	30041073          	csrw	mstatus,s0
1c009626:	409c                	lw	a5,0(s1)
1c009628:	d7ed                	beqz	a5,1c009612 <main+0x74>
1c00962a:	4681                	li	a3,0
1c00962c:	4601                	li	a2,0
1c00962e:	4581                	li	a1,0
1c009630:	4501                	li	a0,0
1c009632:	65d000ef          	jal	ra,1c00a48e <rt_cluster_mount>
1c009636:	40f2                	lw	ra,28(sp)
1c009638:	4462                	lw	s0,24(sp)
1c00963a:	854a                	mv	a0,s2
1c00963c:	44d2                	lw	s1,20(sp)
1c00963e:	4942                	lw	s2,16(sp)
1c009640:	6105                	addi	sp,sp,32
1c009642:	8082                	ret
1c009644:	597d                	li	s2,-1
1c009646:	bfc5                	j	1c009636 <main+0x98>

1c009648 <__rt_event_init>:
1c009648:	02052023          	sw	zero,32(a0)
1c00964c:	02052223          	sw	zero,36(a0)
1c009650:	02052823          	sw	zero,48(a0)
1c009654:	00052023          	sw	zero,0(a0)
1c009658:	8082                	ret

1c00965a <__rt_wait_event_prepare_blocking>:
1c00965a:	00800793          	li	a5,8
1c00965e:	4388                	lw	a0,0(a5)
1c009660:	4d18                	lw	a4,24(a0)
1c009662:	02052223          	sw	zero,36(a0)
1c009666:	c398                	sw	a4,0(a5)
1c009668:	4785                	li	a5,1
1c00966a:	d11c                	sw	a5,32(a0)
1c00966c:	00052023          	sw	zero,0(a0)
1c009670:	8082                	ret

1c009672 <rt_event_alloc>:
1c009672:	1141                	addi	sp,sp,-16
1c009674:	c422                	sw	s0,8(sp)
1c009676:	842e                	mv	s0,a1
1c009678:	c606                	sw	ra,12(sp)
1c00967a:	c226                	sw	s1,4(sp)
1c00967c:	300474f3          	csrrci	s1,mstatus,8
1c009680:	f14027f3          	csrr	a5,mhartid
1c009684:	8795                	srai	a5,a5,0x5
1c009686:	f267b7b3          	p.bclr	a5,a5,25,6
1c00968a:	477d                	li	a4,31
1c00968c:	00378513          	addi	a0,a5,3
1c009690:	00e79363          	bne	a5,a4,1c009696 <rt_event_alloc+0x24>
1c009694:	4501                	li	a0,0
1c009696:	08c00593          	li	a1,140
1c00969a:	02b405b3          	mul	a1,s0,a1
1c00969e:	26ad                	jal	1c009a08 <rt_alloc>
1c0096a0:	87aa                	mv	a5,a0
1c0096a2:	557d                	li	a0,-1
1c0096a4:	cf91                	beqz	a5,1c0096c0 <rt_event_alloc+0x4e>
1c0096a6:	00802683          	lw	a3,8(zero) # 8 <__rt_first_free>
1c0096aa:	4581                	li	a1,0
1c0096ac:	4601                	li	a2,0
1c0096ae:	00800713          	li	a4,8
1c0096b2:	00864c63          	blt	a2,s0,1c0096ca <rt_event_alloc+0x58>
1c0096b6:	c191                	beqz	a1,1c0096ba <rt_event_alloc+0x48>
1c0096b8:	c314                	sw	a3,0(a4)
1c0096ba:	30049073          	csrw	mstatus,s1
1c0096be:	4501                	li	a0,0
1c0096c0:	40b2                	lw	ra,12(sp)
1c0096c2:	4422                	lw	s0,8(sp)
1c0096c4:	4492                	lw	s1,4(sp)
1c0096c6:	0141                	addi	sp,sp,16
1c0096c8:	8082                	ret
1c0096ca:	cf94                	sw	a3,24(a5)
1c0096cc:	0207a023          	sw	zero,32(a5)
1c0096d0:	0207a223          	sw	zero,36(a5)
1c0096d4:	0207a823          	sw	zero,48(a5)
1c0096d8:	0007a023          	sw	zero,0(a5)
1c0096dc:	86be                	mv	a3,a5
1c0096de:	0605                	addi	a2,a2,1
1c0096e0:	4585                	li	a1,1
1c0096e2:	08c78793          	addi	a5,a5,140
1c0096e6:	b7f1                	j	1c0096b2 <rt_event_alloc+0x40>

1c0096e8 <rt_event_get>:
1c0096e8:	30047773          	csrrci	a4,mstatus,8
1c0096ec:	00800793          	li	a5,8
1c0096f0:	4388                	lw	a0,0(a5)
1c0096f2:	c509                	beqz	a0,1c0096fc <rt_event_get+0x14>
1c0096f4:	4d14                	lw	a3,24(a0)
1c0096f6:	c150                	sw	a2,4(a0)
1c0096f8:	c394                	sw	a3,0(a5)
1c0096fa:	c10c                	sw	a1,0(a0)
1c0096fc:	30071073          	csrw	mstatus,a4
1c009700:	8082                	ret

1c009702 <rt_event_get_blocking>:
1c009702:	30047773          	csrrci	a4,mstatus,8
1c009706:	00800793          	li	a5,8
1c00970a:	4388                	lw	a0,0(a5)
1c00970c:	c909                	beqz	a0,1c00971e <rt_event_get_blocking+0x1c>
1c00970e:	4d14                	lw	a3,24(a0)
1c009710:	00052223          	sw	zero,4(a0)
1c009714:	c394                	sw	a3,0(a5)
1c009716:	4785                	li	a5,1
1c009718:	00052023          	sw	zero,0(a0)
1c00971c:	d11c                	sw	a5,32(a0)
1c00971e:	30071073          	csrw	mstatus,a4
1c009722:	8082                	ret

1c009724 <rt_event_push>:
1c009724:	30047773          	csrrci	a4,mstatus,8
1c009728:	00800693          	li	a3,8
1c00972c:	42d4                	lw	a3,4(a3)
1c00972e:	00052c23          	sw	zero,24(a0)
1c009732:	00800793          	li	a5,8
1c009736:	e691                	bnez	a3,1c009742 <rt_event_push+0x1e>
1c009738:	c3c8                	sw	a0,4(a5)
1c00973a:	c788                	sw	a0,8(a5)
1c00973c:	30071073          	csrw	mstatus,a4
1c009740:	8082                	ret
1c009742:	4794                	lw	a3,8(a5)
1c009744:	ce88                	sw	a0,24(a3)
1c009746:	bfd5                	j	1c00973a <rt_event_push+0x16>

1c009748 <__rt_event_execute>:
1c009748:	1141                	addi	sp,sp,-16
1c00974a:	c422                	sw	s0,8(sp)
1c00974c:	00800793          	li	a5,8
1c009750:	43dc                	lw	a5,4(a5)
1c009752:	c606                	sw	ra,12(sp)
1c009754:	c226                	sw	s1,4(sp)
1c009756:	00800413          	li	s0,8
1c00975a:	eb91                	bnez	a5,1c00976e <__rt_event_execute+0x26>
1c00975c:	c1a9                	beqz	a1,1c00979e <__rt_event_execute+0x56>
1c00975e:	10500073          	wfi
1c009762:	30045073          	csrwi	mstatus,8
1c009766:	300477f3          	csrrci	a5,mstatus,8
1c00976a:	405c                	lw	a5,4(s0)
1c00976c:	cb8d                	beqz	a5,1c00979e <__rt_event_execute+0x56>
1c00976e:	4485                	li	s1,1
1c009770:	4f98                	lw	a4,24(a5)
1c009772:	53d4                	lw	a3,36(a5)
1c009774:	00978823          	sb	s1,16(a5)
1c009778:	c058                	sw	a4,4(s0)
1c00977a:	43c8                	lw	a0,4(a5)
1c00977c:	4398                	lw	a4,0(a5)
1c00977e:	e691                	bnez	a3,1c00978a <__rt_event_execute+0x42>
1c009780:	5394                	lw	a3,32(a5)
1c009782:	e681                	bnez	a3,1c00978a <__rt_event_execute+0x42>
1c009784:	4014                	lw	a3,0(s0)
1c009786:	c01c                	sw	a5,0(s0)
1c009788:	cf94                	sw	a3,24(a5)
1c00978a:	0207a023          	sw	zero,32(a5)
1c00978e:	c711                	beqz	a4,1c00979a <__rt_event_execute+0x52>
1c009790:	30045073          	csrwi	mstatus,8
1c009794:	9702                	jalr	a4
1c009796:	300477f3          	csrrci	a5,mstatus,8
1c00979a:	405c                	lw	a5,4(s0)
1c00979c:	fbf1                	bnez	a5,1c009770 <__rt_event_execute+0x28>
1c00979e:	40b2                	lw	ra,12(sp)
1c0097a0:	4422                	lw	s0,8(sp)
1c0097a2:	4492                	lw	s1,4(sp)
1c0097a4:	0141                	addi	sp,sp,16
1c0097a6:	8082                	ret

1c0097a8 <__rt_wait_event>:
1c0097a8:	1141                	addi	sp,sp,-16
1c0097aa:	c422                	sw	s0,8(sp)
1c0097ac:	c606                	sw	ra,12(sp)
1c0097ae:	842a                	mv	s0,a0
1c0097b0:	501c                	lw	a5,32(s0)
1c0097b2:	ef81                	bnez	a5,1c0097ca <__rt_wait_event+0x22>
1c0097b4:	581c                	lw	a5,48(s0)
1c0097b6:	eb91                	bnez	a5,1c0097ca <__rt_wait_event+0x22>
1c0097b8:	00800793          	li	a5,8
1c0097bc:	4398                	lw	a4,0(a5)
1c0097be:	40b2                	lw	ra,12(sp)
1c0097c0:	c380                	sw	s0,0(a5)
1c0097c2:	cc18                	sw	a4,24(s0)
1c0097c4:	4422                	lw	s0,8(sp)
1c0097c6:	0141                	addi	sp,sp,16
1c0097c8:	8082                	ret
1c0097ca:	4585                	li	a1,1
1c0097cc:	4501                	li	a0,0
1c0097ce:	3fad                	jal	1c009748 <__rt_event_execute>
1c0097d0:	b7c5                	j	1c0097b0 <__rt_wait_event+0x8>

1c0097d2 <rt_event_wait>:
1c0097d2:	1141                	addi	sp,sp,-16
1c0097d4:	c606                	sw	ra,12(sp)
1c0097d6:	c422                	sw	s0,8(sp)
1c0097d8:	30047473          	csrrci	s0,mstatus,8
1c0097dc:	37f1                	jal	1c0097a8 <__rt_wait_event>
1c0097de:	30041073          	csrw	mstatus,s0
1c0097e2:	40b2                	lw	ra,12(sp)
1c0097e4:	4422                	lw	s0,8(sp)
1c0097e6:	0141                	addi	sp,sp,16
1c0097e8:	8082                	ret

1c0097ea <__rt_event_sched_init>:
1c0097ea:	00800513          	li	a0,8
1c0097ee:	00052023          	sw	zero,0(a0)
1c0097f2:	00052223          	sw	zero,4(a0)
1c0097f6:	4585                	li	a1,1
1c0097f8:	0511                	addi	a0,a0,4
1c0097fa:	bda5                	j	1c009672 <rt_event_alloc>

1c0097fc <__rt_alloc_account>:
1c0097fc:	01052803          	lw	a6,16(a0)
1c009800:	495c                	lw	a5,20(a0)
1c009802:	4885                	li	a7,1
1c009804:	010898b3          	sll	a7,a7,a6
1c009808:	8d9d                	sub	a1,a1,a5
1c00980a:	411007b3          	neg	a5,a7
1c00980e:	8fed                	and	a5,a5,a1
1c009810:	40b885b3          	sub	a1,a7,a1
1c009814:	0107d833          	srl	a6,a5,a6
1c009818:	95be                	add	a1,a1,a5
1c00981a:	04c5d5b3          	p.minu	a1,a1,a2
1c00981e:	00281e93          	slli	t4,a6,0x2
1c009822:	4781                	li	a5,0
1c009824:	4f05                	li	t5,1
1c009826:	ea15                	bnez	a2,1c00985a <__rt_alloc_account+0x5e>
1c009828:	c3dd                	beqz	a5,1c0098ce <__rt_alloc_account+0xd2>
1c00982a:	02402603          	lw	a2,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c00982e:	c359                	beqz	a4,1c0098b4 <__rt_alloc_account+0xb8>
1c009830:	00479593          	slli	a1,a5,0x4
1c009834:	02802503          	lw	a0,40(zero) # 28 <__rt_alloc_l2_btrim_stdby>
1c009838:	07f6b263          	p.bneimm	a3,-1,1c00989c <__rt_alloc_account+0xa0>
1c00983c:	8dc9                	or	a1,a1,a0
1c00983e:	02b02423          	sw	a1,40(zero) # 28 <__rt_alloc_l2_btrim_stdby>
1c009842:	02802583          	lw	a1,40(zero) # 28 <__rt_alloc_l2_btrim_stdby>
1c009846:	1a104537          	lui	a0,0x1a104
1c00984a:	16b52a23          	sw	a1,372(a0) # 1a104174 <__l1_end+0xa0fd15c>
1c00984e:	07f6b563          	p.bneimm	a3,-1,1c0098b8 <__rt_alloc_account+0xbc>
1c009852:	fff7c793          	not	a5,a5
1c009856:	8e7d                	and	a2,a2,a5
1c009858:	a095                	j	1c0098bc <__rt_alloc_account+0xc0>
1c00985a:	cf15                	beqz	a4,1c009896 <__rt_alloc_account+0x9a>
1c00985c:	00c52303          	lw	t1,12(a0)
1c009860:	9376                	add	t1,t1,t4
1c009862:	00032e03          	lw	t3,0(t1)
1c009866:	01f6b863          	p.bneimm	a3,-1,1c009876 <__rt_alloc_account+0x7a>
1c00986a:	01c89663          	bne	a7,t3,1c009876 <__rt_alloc_account+0x7a>
1c00986e:	010f1fb3          	sll	t6,t5,a6
1c009872:	01f7e7b3          	or	a5,a5,t6
1c009876:	42b68e33          	p.mac	t3,a3,a1
1c00987a:	01c32023          	sw	t3,0(t1)
1c00987e:	011e1663          	bne	t3,a7,1c00988a <__rt_alloc_account+0x8e>
1c009882:	010f1333          	sll	t1,t5,a6
1c009886:	0067e7b3          	or	a5,a5,t1
1c00988a:	8e0d                	sub	a2,a2,a1
1c00988c:	0805                	addi	a6,a6,1
1c00988e:	0e91                	addi	t4,t4,4
1c009890:	04c8d5b3          	p.minu	a1,a7,a2
1c009894:	bf49                	j	1c009826 <__rt_alloc_account+0x2a>
1c009896:	00852303          	lw	t1,8(a0)
1c00989a:	b7d9                	j	1c009860 <__rt_alloc_account+0x64>
1c00989c:	fff5c593          	not	a1,a1
1c0098a0:	8de9                	and	a1,a1,a0
1c0098a2:	bf71                	j	1c00983e <__rt_alloc_account+0x42>
1c0098a4:	07c2                	slli	a5,a5,0x10
1c0098a6:	8e5d                	or	a2,a2,a5
1c0098a8:	a811                	j	1c0098bc <__rt_alloc_account+0xc0>
1c0098aa:	fff7c713          	not	a4,a5
1c0098ae:	8e79                	and	a2,a2,a4
1c0098b0:	07c2                	slli	a5,a5,0x10
1c0098b2:	b745                	j	1c009852 <__rt_alloc_account+0x56>
1c0098b4:	fff6abe3          	p.beqimm	a3,-1,1c0098aa <__rt_alloc_account+0xae>
1c0098b8:	8e5d                	or	a2,a2,a5
1c0098ba:	d76d                	beqz	a4,1c0098a4 <__rt_alloc_account+0xa8>
1c0098bc:	02c02223          	sw	a2,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c0098c0:	02402783          	lw	a5,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c0098c4:	1a104737          	lui	a4,0x1a104
1c0098c8:	18f72023          	sw	a5,384(a4) # 1a104180 <__l1_end+0xa0fd168>
1c0098cc:	8082                	ret
1c0098ce:	8082                	ret

1c0098d0 <__rt_alloc_account_alloc>:
1c0098d0:	415c                	lw	a5,4(a0)
1c0098d2:	c781                	beqz	a5,1c0098da <__rt_alloc_account_alloc+0xa>
1c0098d4:	4701                	li	a4,0
1c0098d6:	56fd                	li	a3,-1
1c0098d8:	b715                	j	1c0097fc <__rt_alloc_account>
1c0098da:	8082                	ret

1c0098dc <__rt_alloc_account_free>:
1c0098dc:	415c                	lw	a5,4(a0)
1c0098de:	c781                	beqz	a5,1c0098e6 <__rt_alloc_account_free+0xa>
1c0098e0:	4701                	li	a4,0
1c0098e2:	4685                	li	a3,1
1c0098e4:	bf21                	j	1c0097fc <__rt_alloc_account>
1c0098e6:	8082                	ret

1c0098e8 <rt_user_alloc_init>:
1c0098e8:	00758793          	addi	a5,a1,7
1c0098ec:	c407b7b3          	p.bclr	a5,a5,2,0
1c0098f0:	40b785b3          	sub	a1,a5,a1
1c0098f4:	00052223          	sw	zero,4(a0)
1c0098f8:	c11c                	sw	a5,0(a0)
1c0098fa:	8e0d                	sub	a2,a2,a1
1c0098fc:	00c05763          	blez	a2,1c00990a <rt_user_alloc_init+0x22>
1c009900:	c4063633          	p.bclr	a2,a2,2,0
1c009904:	c390                	sw	a2,0(a5)
1c009906:	0007a223          	sw	zero,4(a5)
1c00990a:	8082                	ret

1c00990c <rt_user_alloc>:
1c00990c:	1141                	addi	sp,sp,-16
1c00990e:	c422                	sw	s0,8(sp)
1c009910:	4100                	lw	s0,0(a0)
1c009912:	059d                	addi	a1,a1,7
1c009914:	c606                	sw	ra,12(sp)
1c009916:	c226                	sw	s1,4(sp)
1c009918:	c04a                	sw	s2,0(sp)
1c00991a:	c405b7b3          	p.bclr	a5,a1,2,0
1c00991e:	4701                	li	a4,0
1c009920:	cc19                	beqz	s0,1c00993e <rt_user_alloc+0x32>
1c009922:	4010                	lw	a2,0(s0)
1c009924:	4054                	lw	a3,4(s0)
1c009926:	02f64363          	blt	a2,a5,1c00994c <rt_user_alloc+0x40>
1c00992a:	84aa                	mv	s1,a0
1c00992c:	00840593          	addi	a1,s0,8
1c009930:	02f61363          	bne	a2,a5,1c009956 <rt_user_alloc+0x4a>
1c009934:	cf19                	beqz	a4,1c009952 <rt_user_alloc+0x46>
1c009936:	c354                	sw	a3,4(a4)
1c009938:	1661                	addi	a2,a2,-8
1c00993a:	8526                	mv	a0,s1
1c00993c:	3f51                	jal	1c0098d0 <__rt_alloc_account_alloc>
1c00993e:	8522                	mv	a0,s0
1c009940:	40b2                	lw	ra,12(sp)
1c009942:	4422                	lw	s0,8(sp)
1c009944:	4492                	lw	s1,4(sp)
1c009946:	4902                	lw	s2,0(sp)
1c009948:	0141                	addi	sp,sp,16
1c00994a:	8082                	ret
1c00994c:	8722                	mv	a4,s0
1c00994e:	8436                	mv	s0,a3
1c009950:	bfc1                	j	1c009920 <rt_user_alloc+0x14>
1c009952:	c094                	sw	a3,0(s1)
1c009954:	b7d5                	j	1c009938 <rt_user_alloc+0x2c>
1c009956:	00f40933          	add	s2,s0,a5
1c00995a:	8e1d                	sub	a2,a2,a5
1c00995c:	00c92023          	sw	a2,0(s2)
1c009960:	00d92223          	sw	a3,4(s2)
1c009964:	cb11                	beqz	a4,1c009978 <rt_user_alloc+0x6c>
1c009966:	01272223          	sw	s2,4(a4)
1c00996a:	ff878613          	addi	a2,a5,-8
1c00996e:	8526                	mv	a0,s1
1c009970:	3785                	jal	1c0098d0 <__rt_alloc_account_alloc>
1c009972:	4621                	li	a2,8
1c009974:	85ca                	mv	a1,s2
1c009976:	b7d1                	j	1c00993a <rt_user_alloc+0x2e>
1c009978:	0124a023          	sw	s2,0(s1)
1c00997c:	b7fd                	j	1c00996a <rt_user_alloc+0x5e>

1c00997e <rt_user_free>:
1c00997e:	1101                	addi	sp,sp,-32
1c009980:	cc22                	sw	s0,24(sp)
1c009982:	842e                	mv	s0,a1
1c009984:	410c                	lw	a1,0(a0)
1c009986:	061d                	addi	a2,a2,7
1c009988:	ca26                	sw	s1,20(sp)
1c00998a:	c84a                	sw	s2,16(sp)
1c00998c:	c64e                	sw	s3,12(sp)
1c00998e:	ce06                	sw	ra,28(sp)
1c009990:	89aa                	mv	s3,a0
1c009992:	c40634b3          	p.bclr	s1,a2,2,0
1c009996:	4901                	li	s2,0
1c009998:	c199                	beqz	a1,1c00999e <rt_user_free+0x20>
1c00999a:	0485e763          	bltu	a1,s0,1c0099e8 <rt_user_free+0x6a>
1c00999e:	009407b3          	add	a5,s0,s1
1c0099a2:	04f59663          	bne	a1,a5,1c0099ee <rt_user_free+0x70>
1c0099a6:	419c                	lw	a5,0(a1)
1c0099a8:	4621                	li	a2,8
1c0099aa:	854e                	mv	a0,s3
1c0099ac:	97a6                	add	a5,a5,s1
1c0099ae:	c01c                	sw	a5,0(s0)
1c0099b0:	41dc                	lw	a5,4(a1)
1c0099b2:	c05c                	sw	a5,4(s0)
1c0099b4:	3725                	jal	1c0098dc <__rt_alloc_account_free>
1c0099b6:	04090663          	beqz	s2,1c009a02 <rt_user_free+0x84>
1c0099ba:	00092703          	lw	a4,0(s2)
1c0099be:	00e907b3          	add	a5,s2,a4
1c0099c2:	02f41963          	bne	s0,a5,1c0099f4 <rt_user_free+0x76>
1c0099c6:	401c                	lw	a5,0(s0)
1c0099c8:	8626                	mv	a2,s1
1c0099ca:	85a2                	mv	a1,s0
1c0099cc:	97ba                	add	a5,a5,a4
1c0099ce:	00f92023          	sw	a5,0(s2)
1c0099d2:	405c                	lw	a5,4(s0)
1c0099d4:	00f92223          	sw	a5,4(s2)
1c0099d8:	4462                	lw	s0,24(sp)
1c0099da:	40f2                	lw	ra,28(sp)
1c0099dc:	44d2                	lw	s1,20(sp)
1c0099de:	4942                	lw	s2,16(sp)
1c0099e0:	854e                	mv	a0,s3
1c0099e2:	49b2                	lw	s3,12(sp)
1c0099e4:	6105                	addi	sp,sp,32
1c0099e6:	bddd                	j	1c0098dc <__rt_alloc_account_free>
1c0099e8:	892e                	mv	s2,a1
1c0099ea:	41cc                	lw	a1,4(a1)
1c0099ec:	b775                	j	1c009998 <rt_user_free+0x1a>
1c0099ee:	c004                	sw	s1,0(s0)
1c0099f0:	c04c                	sw	a1,4(s0)
1c0099f2:	b7d1                	j	1c0099b6 <rt_user_free+0x38>
1c0099f4:	00892223          	sw	s0,4(s2)
1c0099f8:	ff848613          	addi	a2,s1,-8
1c0099fc:	00840593          	addi	a1,s0,8
1c009a00:	bfe1                	j	1c0099d8 <rt_user_free+0x5a>
1c009a02:	0089a023          	sw	s0,0(s3)
1c009a06:	bfcd                	j	1c0099f8 <rt_user_free+0x7a>

1c009a08 <rt_alloc>:
1c009a08:	1101                	addi	sp,sp,-32
1c009a0a:	cc22                	sw	s0,24(sp)
1c009a0c:	ce06                	sw	ra,28(sp)
1c009a0e:	ca26                	sw	s1,20(sp)
1c009a10:	c84a                	sw	s2,16(sp)
1c009a12:	c64e                	sw	s3,12(sp)
1c009a14:	c452                	sw	s4,8(sp)
1c009a16:	4789                	li	a5,2
1c009a18:	842a                	mv	s0,a0
1c009a1a:	02a7ed63          	bltu	a5,a0,1c009a54 <rt_alloc+0x4c>
1c009a1e:	1c001937          	lui	s2,0x1c001
1c009a22:	89ae                	mv	s3,a1
1c009a24:	448d                	li	s1,3
1c009a26:	4a61                	li	s4,24
1c009a28:	76490913          	addi	s2,s2,1892 # 1c001764 <__rt_alloc_l2>
1c009a2c:	854a                	mv	a0,s2
1c009a2e:	43440533          	p.mac	a0,s0,s4
1c009a32:	85ce                	mv	a1,s3
1c009a34:	3de1                	jal	1c00990c <rt_user_alloc>
1c009a36:	e519                	bnez	a0,1c009a44 <rt_alloc+0x3c>
1c009a38:	0405                	addi	s0,s0,1
1c009a3a:	00343363          	p.bneimm	s0,3,1c009a40 <rt_alloc+0x38>
1c009a3e:	4401                	li	s0,0
1c009a40:	14fd                	addi	s1,s1,-1
1c009a42:	f4ed                	bnez	s1,1c009a2c <rt_alloc+0x24>
1c009a44:	40f2                	lw	ra,28(sp)
1c009a46:	4462                	lw	s0,24(sp)
1c009a48:	44d2                	lw	s1,20(sp)
1c009a4a:	4942                	lw	s2,16(sp)
1c009a4c:	49b2                	lw	s3,12(sp)
1c009a4e:	4a22                	lw	s4,8(sp)
1c009a50:	6105                	addi	sp,sp,32
1c009a52:	8082                	ret
1c009a54:	1c0017b7          	lui	a5,0x1c001
1c009a58:	ffd50413          	addi	s0,a0,-3
1c009a5c:	7607a503          	lw	a0,1888(a5) # 1c001760 <__rt_alloc_l1>
1c009a60:	47e1                	li	a5,24
1c009a62:	40f2                	lw	ra,28(sp)
1c009a64:	42f40533          	p.mac	a0,s0,a5
1c009a68:	4462                	lw	s0,24(sp)
1c009a6a:	44d2                	lw	s1,20(sp)
1c009a6c:	4942                	lw	s2,16(sp)
1c009a6e:	49b2                	lw	s3,12(sp)
1c009a70:	4a22                	lw	s4,8(sp)
1c009a72:	6105                	addi	sp,sp,32
1c009a74:	bd61                	j	1c00990c <rt_user_alloc>

1c009a76 <__rt_alloc_init_l1>:
1c009a76:	1c0017b7          	lui	a5,0x1c001
1c009a7a:	7607a703          	lw	a4,1888(a5) # 1c001760 <__rt_alloc_l1>
1c009a7e:	100077b7          	lui	a5,0x10007
1c009a82:	01651593          	slli	a1,a0,0x16
1c009a86:	01878793          	addi	a5,a5,24 # 10007018 <__l1_end>
1c009a8a:	95be                	add	a1,a1,a5
1c009a8c:	47e1                	li	a5,24
1c009a8e:	42f50733          	p.mac	a4,a0,a5
1c009a92:	6665                	lui	a2,0x19
1c009a94:	fe860613          	addi	a2,a2,-24 # 18fe8 <__l1_heap_size>
1c009a98:	853a                	mv	a0,a4
1c009a9a:	b5b9                	j	1c0098e8 <rt_user_alloc_init>

1c009a9c <__rt_alloc_init_l1_for_fc>:
1c009a9c:	100075b7          	lui	a1,0x10007
1c009aa0:	01651793          	slli	a5,a0,0x16
1c009aa4:	01858593          	addi	a1,a1,24 # 10007018 <__l1_end>
1c009aa8:	00b78733          	add	a4,a5,a1
1c009aac:	07e1                	addi	a5,a5,24
1c009aae:	1c0016b7          	lui	a3,0x1c001
1c009ab2:	95be                	add	a1,a1,a5
1c009ab4:	47e1                	li	a5,24
1c009ab6:	76e6a023          	sw	a4,1888(a3) # 1c001760 <__rt_alloc_l1>
1c009aba:	42f50733          	p.mac	a4,a0,a5
1c009abe:	6665                	lui	a2,0x19
1c009ac0:	fd060613          	addi	a2,a2,-48 # 18fd0 <_l1_preload_size+0x11fc0>
1c009ac4:	853a                	mv	a0,a4
1c009ac6:	b50d                	j	1c0098e8 <rt_user_alloc_init>

1c009ac8 <__rt_allocs_init>:
1c009ac8:	1141                	addi	sp,sp,-16
1c009aca:	1c0015b7          	lui	a1,0x1c001
1c009ace:	c606                	sw	ra,12(sp)
1c009ad0:	c422                	sw	s0,8(sp)
1c009ad2:	c226                	sw	s1,4(sp)
1c009ad4:	c04a                	sw	s2,0(sp)
1c009ad6:	7f858793          	addi	a5,a1,2040 # 1c0017f8 <__l2_priv0_end>
1c009ada:	1c008637          	lui	a2,0x1c008
1c009ade:	0cc7c863          	blt	a5,a2,1c009bae <__rt_allocs_init+0xe6>
1c009ae2:	4581                	li	a1,0
1c009ae4:	4601                	li	a2,0
1c009ae6:	1c001437          	lui	s0,0x1c001
1c009aea:	76440513          	addi	a0,s0,1892 # 1c001764 <__rt_alloc_l2>
1c009aee:	3bed                	jal	1c0098e8 <rt_user_alloc_init>
1c009af0:	1c00c5b7          	lui	a1,0x1c00c
1c009af4:	5f858793          	addi	a5,a1,1528 # 1c00c5f8 <__l2_priv1_end>
1c009af8:	1c010637          	lui	a2,0x1c010
1c009afc:	0ac7cd63          	blt	a5,a2,1c009bb6 <__rt_allocs_init+0xee>
1c009b00:	4581                	li	a1,0
1c009b02:	4601                	li	a2,0
1c009b04:	1c001537          	lui	a0,0x1c001
1c009b08:	77c50513          	addi	a0,a0,1916 # 1c00177c <__rt_alloc_l2+0x18>
1c009b0c:	3bf1                	jal	1c0098e8 <rt_user_alloc_init>
1c009b0e:	1c0175b7          	lui	a1,0x1c017
1c009b12:	1a058793          	addi	a5,a1,416 # 1c0171a0 <__l2_shared_end>
1c009b16:	1c190937          	lui	s2,0x1c190
1c009b1a:	40f90933          	sub	s2,s2,a5
1c009b1e:	1c0014b7          	lui	s1,0x1c001
1c009b22:	864a                	mv	a2,s2
1c009b24:	1a058593          	addi	a1,a1,416
1c009b28:	79448513          	addi	a0,s1,1940 # 1c001794 <__rt_alloc_l2+0x30>
1c009b2c:	3b75                	jal	1c0098e8 <rt_user_alloc_init>
1c009b2e:	76440413          	addi	s0,s0,1892
1c009b32:	4785                	li	a5,1
1c009b34:	d85c                	sw	a5,52(s0)
1c009b36:	03000593          	li	a1,48
1c009b3a:	4501                	li	a0,0
1c009b3c:	35f1                	jal	1c009a08 <rt_alloc>
1c009b3e:	dc08                	sw	a0,56(s0)
1c009b40:	03000593          	li	a1,48
1c009b44:	4501                	li	a0,0
1c009b46:	35c9                	jal	1c009a08 <rt_alloc>
1c009b48:	5c14                	lw	a3,56(s0)
1c009b4a:	1c0017b7          	lui	a5,0x1c001
1c009b4e:	dc48                	sw	a0,60(s0)
1c009b50:	76478793          	addi	a5,a5,1892 # 1c001764 <__rt_alloc_l2>
1c009b54:	00c250fb          	lp.setupi	x1,12,1c009b5c <__rt_allocs_init+0x94>
1c009b58:	0006a22b          	p.sw	zero,4(a3!)
1c009b5c:	0005222b          	p.sw	zero,4(a0!)
1c009b60:	4745                	li	a4,17
1c009b62:	c3b8                	sw	a4,64(a5)
1c009b64:	1c0175b7          	lui	a1,0x1c017
1c009b68:	1c010737          	lui	a4,0x1c010
1c009b6c:	c3f8                	sw	a4,68(a5)
1c009b6e:	00890613          	addi	a2,s2,8 # 1c190008 <__l2_shared_end+0x178e68>
1c009b72:	19858593          	addi	a1,a1,408 # 1c017198 <_l1_preload_start_inL2+0x7008>
1c009b76:	79448513          	addi	a0,s1,1940
1c009b7a:	338d                	jal	1c0098dc <__rt_alloc_account_free>
1c009b7c:	f14027f3          	csrr	a5,mhartid
1c009b80:	ca5797b3          	p.extractu	a5,a5,5,5
1c009b84:	eb81                	bnez	a5,1c009b94 <__rt_allocs_init+0xcc>
1c009b86:	4422                	lw	s0,8(sp)
1c009b88:	40b2                	lw	ra,12(sp)
1c009b8a:	4492                	lw	s1,4(sp)
1c009b8c:	4902                	lw	s2,0(sp)
1c009b8e:	4501                	li	a0,0
1c009b90:	0141                	addi	sp,sp,16
1c009b92:	b729                	j	1c009a9c <__rt_alloc_init_l1_for_fc>
1c009b94:	45e1                	li	a1,24
1c009b96:	4501                	li	a0,0
1c009b98:	3d85                	jal	1c009a08 <rt_alloc>
1c009b9a:	40b2                	lw	ra,12(sp)
1c009b9c:	4422                	lw	s0,8(sp)
1c009b9e:	1c0017b7          	lui	a5,0x1c001
1c009ba2:	76a7a023          	sw	a0,1888(a5) # 1c001760 <__rt_alloc_l1>
1c009ba6:	4492                	lw	s1,4(sp)
1c009ba8:	4902                	lw	s2,0(sp)
1c009baa:	0141                	addi	sp,sp,16
1c009bac:	8082                	ret
1c009bae:	8e1d                	sub	a2,a2,a5
1c009bb0:	7f858593          	addi	a1,a1,2040
1c009bb4:	bf0d                	j	1c009ae6 <__rt_allocs_init+0x1e>
1c009bb6:	8e1d                	sub	a2,a2,a5
1c009bb8:	5f858593          	addi	a1,a1,1528
1c009bbc:	b7a1                	j	1c009b04 <__rt_allocs_init+0x3c>

1c009bbe <__rt_time_poweroff>:
1c009bbe:	1a10b7b7          	lui	a5,0x1a10b
1c009bc2:	0791                	addi	a5,a5,4
1c009bc4:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa103ff0>
1c009bc8:	1c001737          	lui	a4,0x1c001
1c009bcc:	72f72e23          	sw	a5,1852(a4) # 1c00173c <timer_count>
1c009bd0:	4501                	li	a0,0
1c009bd2:	8082                	ret

1c009bd4 <__rt_time_poweron>:
1c009bd4:	1c0017b7          	lui	a5,0x1c001
1c009bd8:	73c7a703          	lw	a4,1852(a5) # 1c00173c <timer_count>
1c009bdc:	1a10b7b7          	lui	a5,0x1a10b
1c009be0:	0791                	addi	a5,a5,4
1c009be2:	00e7a423          	sw	a4,8(a5) # 1a10b008 <__l1_end+0xa103ff0>
1c009be6:	4501                	li	a0,0
1c009be8:	8082                	ret

1c009bea <rt_event_push_delayed>:
1c009bea:	30047373          	csrrci	t1,mstatus,8
1c009bee:	1c001637          	lui	a2,0x1c001
1c009bf2:	7ac62703          	lw	a4,1964(a2) # 1c0017ac <first_delayed>
1c009bf6:	1a10b7b7          	lui	a5,0x1a10b
1c009bfa:	0791                	addi	a5,a5,4
1c009bfc:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa103ff0>
1c009c00:	46f9                	li	a3,30
1c009c02:	0405e5b3          	p.max	a1,a1,zero
1c009c06:	02d5c5b3          	div	a1,a1,a3
1c009c0a:	800006b7          	lui	a3,0x80000
1c009c0e:	fff6c693          	not	a3,a3
1c009c12:	00d7f833          	and	a6,a5,a3
1c009c16:	0585                	addi	a1,a1,1
1c009c18:	97ae                	add	a5,a5,a1
1c009c1a:	d95c                	sw	a5,52(a0)
1c009c1c:	982e                	add	a6,a6,a1
1c009c1e:	4781                	li	a5,0
1c009c20:	c719                	beqz	a4,1c009c2e <rt_event_push_delayed+0x44>
1c009c22:	03472883          	lw	a7,52(a4)
1c009c26:	00d8f8b3          	and	a7,a7,a3
1c009c2a:	0108e863          	bltu	a7,a6,1c009c3a <rt_event_push_delayed+0x50>
1c009c2e:	cb89                	beqz	a5,1c009c40 <rt_event_push_delayed+0x56>
1c009c30:	cf88                	sw	a0,24(a5)
1c009c32:	cd18                	sw	a4,24(a0)
1c009c34:	30031073          	csrw	mstatus,t1
1c009c38:	8082                	ret
1c009c3a:	87ba                	mv	a5,a4
1c009c3c:	4f18                	lw	a4,24(a4)
1c009c3e:	b7cd                	j	1c009c20 <rt_event_push_delayed+0x36>
1c009c40:	1a10b7b7          	lui	a5,0x1a10b
1c009c44:	0791                	addi	a5,a5,4
1c009c46:	7aa62623          	sw	a0,1964(a2)
1c009c4a:	cd18                	sw	a4,24(a0)
1c009c4c:	0087a703          	lw	a4,8(a5) # 1a10b008 <__l1_end+0xa103ff0>
1c009c50:	95ba                	add	a1,a1,a4
1c009c52:	00b7a823          	sw	a1,16(a5)
1c009c56:	08500713          	li	a4,133
1c009c5a:	00e7a023          	sw	a4,0(a5)
1c009c5e:	bfd9                	j	1c009c34 <rt_event_push_delayed+0x4a>

1c009c60 <rt_time_wait_us>:
1c009c60:	1101                	addi	sp,sp,-32
1c009c62:	85aa                	mv	a1,a0
1c009c64:	4501                	li	a0,0
1c009c66:	ce06                	sw	ra,28(sp)
1c009c68:	cc22                	sw	s0,24(sp)
1c009c6a:	c62e                	sw	a1,12(sp)
1c009c6c:	3c59                	jal	1c009702 <rt_event_get_blocking>
1c009c6e:	45b2                	lw	a1,12(sp)
1c009c70:	842a                	mv	s0,a0
1c009c72:	3fa5                	jal	1c009bea <rt_event_push_delayed>
1c009c74:	8522                	mv	a0,s0
1c009c76:	4462                	lw	s0,24(sp)
1c009c78:	40f2                	lw	ra,28(sp)
1c009c7a:	6105                	addi	sp,sp,32
1c009c7c:	be99                	j	1c0097d2 <rt_event_wait>

1c009c7e <__rt_time_init>:
1c009c7e:	1c0017b7          	lui	a5,0x1c001
1c009c82:	7a07a623          	sw	zero,1964(a5) # 1c0017ac <first_delayed>
1c009c86:	1a10b7b7          	lui	a5,0x1a10b
1c009c8a:	1141                	addi	sp,sp,-16
1c009c8c:	08300713          	li	a4,131
1c009c90:	0791                	addi	a5,a5,4
1c009c92:	c606                	sw	ra,12(sp)
1c009c94:	c422                	sw	s0,8(sp)
1c009c96:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa103fe8>
1c009c9a:	1c00a5b7          	lui	a1,0x1c00a
1c009c9e:	d1858593          	addi	a1,a1,-744 # 1c009d18 <__rt_timer_handler>
1c009ca2:	452d                	li	a0,11
1c009ca4:	539000ef          	jal	ra,1c00a9dc <rt_irq_set_handler>
1c009ca8:	6785                	lui	a5,0x1
1c009caa:	f1402773          	csrr	a4,mhartid
1c009cae:	46fd                	li	a3,31
1c009cb0:	ca571733          	p.extractu	a4,a4,5,5
1c009cb4:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c009cb8:	04d71863          	bne	a4,a3,1c009d08 <__rt_time_init+0x8a>
1c009cbc:	1a109737          	lui	a4,0x1a109
1c009cc0:	c35c                	sw	a5,4(a4)
1c009cc2:	1c00a5b7          	lui	a1,0x1c00a
1c009cc6:	4601                	li	a2,0
1c009cc8:	bbe58593          	addi	a1,a1,-1090 # 1c009bbe <__rt_time_poweroff>
1c009ccc:	4509                	li	a0,2
1c009cce:	69d000ef          	jal	ra,1c00ab6a <__rt_cbsys_add>
1c009cd2:	1c00a5b7          	lui	a1,0x1c00a
1c009cd6:	842a                	mv	s0,a0
1c009cd8:	4601                	li	a2,0
1c009cda:	bd458593          	addi	a1,a1,-1068 # 1c009bd4 <__rt_time_poweron>
1c009cde:	450d                	li	a0,3
1c009ce0:	68b000ef          	jal	ra,1c00ab6a <__rt_cbsys_add>
1c009ce4:	8d41                	or	a0,a0,s0
1c009ce6:	c50d                	beqz	a0,1c009d10 <__rt_time_init+0x92>
1c009ce8:	f1402673          	csrr	a2,mhartid
1c009cec:	1c001537          	lui	a0,0x1c001
1c009cf0:	40565593          	srai	a1,a2,0x5
1c009cf4:	f265b5b3          	p.bclr	a1,a1,25,6
1c009cf8:	f4563633          	p.bclr	a2,a2,26,5
1c009cfc:	a3450513          	addi	a0,a0,-1484 # 1c000a34 <PIo2+0xd8>
1c009d00:	467010ef          	jal	ra,1c00b966 <printf>
1c009d04:	3f1010ef          	jal	ra,1c00b8f4 <abort>
1c009d08:	00204737          	lui	a4,0x204
1c009d0c:	cb5c                	sw	a5,20(a4)
1c009d0e:	bf55                	j	1c009cc2 <__rt_time_init+0x44>
1c009d10:	40b2                	lw	ra,12(sp)
1c009d12:	4422                	lw	s0,8(sp)
1c009d14:	0141                	addi	sp,sp,16
1c009d16:	8082                	ret

1c009d18 <__rt_timer_handler>:
1c009d18:	7179                	addi	sp,sp,-48
1c009d1a:	ce36                	sw	a3,28(sp)
1c009d1c:	1c0016b7          	lui	a3,0x1c001
1c009d20:	ca3e                	sw	a5,20(sp)
1c009d22:	7ac6a783          	lw	a5,1964(a3) # 1c0017ac <first_delayed>
1c009d26:	cc3a                	sw	a4,24(sp)
1c009d28:	1a10b737          	lui	a4,0x1a10b
1c009d2c:	0711                	addi	a4,a4,4
1c009d2e:	d61a                	sw	t1,44(sp)
1c009d30:	d42a                	sw	a0,40(sp)
1c009d32:	d22e                	sw	a1,36(sp)
1c009d34:	d032                	sw	a2,32(sp)
1c009d36:	c842                	sw	a6,16(sp)
1c009d38:	c646                	sw	a7,12(sp)
1c009d3a:	00872703          	lw	a4,8(a4) # 1a10b008 <__l1_end+0xa103ff0>
1c009d3e:	00c02583          	lw	a1,12(zero) # c <__rt_sched>
1c009d42:	01002603          	lw	a2,16(zero) # 10 <__rt_sched+0x4>
1c009d46:	800008b7          	lui	a7,0x80000
1c009d4a:	4501                	li	a0,0
1c009d4c:	4801                	li	a6,0
1c009d4e:	ffe8c893          	xori	a7,a7,-2
1c009d52:	e3a5                	bnez	a5,1c009db2 <__rt_timer_handler+0x9a>
1c009d54:	00080463          	beqz	a6,1c009d5c <__rt_timer_handler+0x44>
1c009d58:	00b02623          	sw	a1,12(zero) # c <__rt_sched>
1c009d5c:	c119                	beqz	a0,1c009d62 <__rt_timer_handler+0x4a>
1c009d5e:	00c02823          	sw	a2,16(zero) # 10 <__rt_sched+0x4>
1c009d62:	1a10b7b7          	lui	a5,0x1a10b
1c009d66:	08100713          	li	a4,129
1c009d6a:	0791                	addi	a5,a5,4
1c009d6c:	7a06a623          	sw	zero,1964(a3)
1c009d70:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa103fe8>
1c009d74:	6785                	lui	a5,0x1
1c009d76:	1a109737          	lui	a4,0x1a109
1c009d7a:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c009d7e:	cb5c                	sw	a5,20(a4)
1c009d80:	5332                	lw	t1,44(sp)
1c009d82:	5522                	lw	a0,40(sp)
1c009d84:	5592                	lw	a1,36(sp)
1c009d86:	5602                	lw	a2,32(sp)
1c009d88:	46f2                	lw	a3,28(sp)
1c009d8a:	4762                	lw	a4,24(sp)
1c009d8c:	47d2                	lw	a5,20(sp)
1c009d8e:	4842                	lw	a6,16(sp)
1c009d90:	48b2                	lw	a7,12(sp)
1c009d92:	6145                	addi	sp,sp,48
1c009d94:	30200073          	mret
1c009d98:	0187a303          	lw	t1,24(a5)
1c009d9c:	0007ac23          	sw	zero,24(a5)
1c009da0:	c591                	beqz	a1,1c009dac <__rt_timer_handler+0x94>
1c009da2:	ce1c                	sw	a5,24(a2)
1c009da4:	863e                	mv	a2,a5
1c009da6:	4505                	li	a0,1
1c009da8:	879a                	mv	a5,t1
1c009daa:	b765                	j	1c009d52 <__rt_timer_handler+0x3a>
1c009dac:	85be                	mv	a1,a5
1c009dae:	4805                	li	a6,1
1c009db0:	bfd5                	j	1c009da4 <__rt_timer_handler+0x8c>
1c009db2:	0347a303          	lw	t1,52(a5)
1c009db6:	40670333          	sub	t1,a4,t1
1c009dba:	fc68ffe3          	bleu	t1,a7,1c009d98 <__rt_timer_handler+0x80>
1c009dbe:	00080463          	beqz	a6,1c009dc6 <__rt_timer_handler+0xae>
1c009dc2:	00b02623          	sw	a1,12(zero) # c <__rt_sched>
1c009dc6:	c119                	beqz	a0,1c009dcc <__rt_timer_handler+0xb4>
1c009dc8:	00c02823          	sw	a2,16(zero) # 10 <__rt_sched+0x4>
1c009dcc:	7af6a623          	sw	a5,1964(a3)
1c009dd0:	1a10b6b7          	lui	a3,0x1a10b
1c009dd4:	0691                	addi	a3,a3,4
1c009dd6:	0086a603          	lw	a2,8(a3) # 1a10b008 <__l1_end+0xa103ff0>
1c009dda:	5bdc                	lw	a5,52(a5)
1c009ddc:	40e78733          	sub	a4,a5,a4
1c009de0:	9732                	add	a4,a4,a2
1c009de2:	00e6a823          	sw	a4,16(a3)
1c009de6:	08500793          	li	a5,133
1c009dea:	00f6a023          	sw	a5,0(a3)
1c009dee:	bf49                	j	1c009d80 <__rt_timer_handler+0x68>

1c009df0 <__rt_pmu_change_domain_power>:
1c009df0:	1c0017b7          	lui	a5,0x1c001
1c009df4:	4607a883          	lw	a7,1120(a5) # 1c001460 <stack>
1c009df8:	ffd60813          	addi	a6,a2,-3
1c009dfc:	46078793          	addi	a5,a5,1120
1c009e00:	06089563          	bnez	a7,1c009e6a <__rt_pmu_change_domain_power+0x7a>
1c009e04:	4585                	li	a1,1
1c009e06:	0105f463          	bleu	a6,a1,1c009e0e <__rt_pmu_change_domain_power+0x1e>
1c009e0a:	02a02023          	sw	a0,32(zero) # 20 <__rt_pmu_scu_event>
1c009e0e:	0047a803          	lw	a6,4(a5)
1c009e12:	4505                	li	a0,1
1c009e14:	00c515b3          	sll	a1,a0,a2
1c009e18:	fff5c593          	not	a1,a1
1c009e1c:	0105f5b3          	and	a1,a1,a6
1c009e20:	00c69833          	sll	a6,a3,a2
1c009e24:	0105e5b3          	or	a1,a1,a6
1c009e28:	c3cc                	sw	a1,4(a5)
1c009e2a:	c388                	sw	a0,0(a5)
1c009e2c:	02463b63          	p.bneimm	a2,4,1c009e62 <__rt_pmu_change_domain_power+0x72>
1c009e30:	fc1737b3          	p.bclr	a5,a4,30,1
1c009e34:	8b09                	andi	a4,a4,2
1c009e36:	00478693          	addi	a3,a5,4
1c009e3a:	c319                	beqz	a4,1c009e40 <__rt_pmu_change_domain_power+0x50>
1c009e3c:	00e78693          	addi	a3,a5,14
1c009e40:	0036d793          	srli	a5,a3,0x3
1c009e44:	6741                	lui	a4,0x10
1c009e46:	0789                	addi	a5,a5,2
1c009e48:	f836b6b3          	p.bclr	a3,a3,28,3
1c009e4c:	0786                	slli	a5,a5,0x1
1c009e4e:	00d716b3          	sll	a3,a4,a3
1c009e52:	8edd                	or	a3,a3,a5
1c009e54:	0416e693          	ori	a3,a3,65
1c009e58:	1a1077b7          	lui	a5,0x1a107
1c009e5c:	00d7a023          	sw	a3,0(a5) # 1a107000 <__l1_end+0xa0fffe8>
1c009e60:	8082                	ret
1c009e62:	060d                	addi	a2,a2,3
1c009e64:	0606                	slli	a2,a2,0x1
1c009e66:	96b2                	add	a3,a3,a2
1c009e68:	bfe1                	j	1c009e40 <__rt_pmu_change_domain_power+0x50>
1c009e6a:	dd58                	sw	a4,60(a0)
1c009e6c:	00283813          	sltiu	a6,a6,2
1c009e70:	4798                	lw	a4,8(a5)
1c009e72:	00184813          	xori	a6,a6,1
1c009e76:	d950                	sw	a2,52(a0)
1c009e78:	dd14                	sw	a3,56(a0)
1c009e7a:	05052023          	sw	a6,64(a0)
1c009e7e:	eb09                	bnez	a4,1c009e90 <__rt_pmu_change_domain_power+0xa0>
1c009e80:	c788                	sw	a0,8(a5)
1c009e82:	c7c8                	sw	a0,12(a5)
1c009e84:	00052c23          	sw	zero,24(a0)
1c009e88:	c199                	beqz	a1,1c009e8e <__rt_pmu_change_domain_power+0x9e>
1c009e8a:	4785                	li	a5,1
1c009e8c:	c19c                	sw	a5,0(a1)
1c009e8e:	8082                	ret
1c009e90:	47d8                	lw	a4,12(a5)
1c009e92:	cf08                	sw	a0,24(a4)
1c009e94:	b7fd                	j	1c009e82 <__rt_pmu_change_domain_power+0x92>

1c009e96 <__rt_pmu_cluster_power_down>:
1c009e96:	4701                	li	a4,0
1c009e98:	4681                	li	a3,0
1c009e9a:	460d                	li	a2,3
1c009e9c:	bf91                	j	1c009df0 <__rt_pmu_change_domain_power>

1c009e9e <__rt_pmu_cluster_power_up>:
1c009e9e:	1141                	addi	sp,sp,-16
1c009ea0:	4701                	li	a4,0
1c009ea2:	4685                	li	a3,1
1c009ea4:	460d                	li	a2,3
1c009ea6:	c606                	sw	ra,12(sp)
1c009ea8:	37a1                	jal	1c009df0 <__rt_pmu_change_domain_power>
1c009eaa:	40b2                	lw	ra,12(sp)
1c009eac:	02002223          	sw	zero,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c009eb0:	03000713          	li	a4,48
1c009eb4:	1c0017b7          	lui	a5,0x1c001
1c009eb8:	46e7a823          	sw	a4,1136(a5) # 1c001470 <__rt_alloc_l1_pwr_ctrl>
1c009ebc:	4505                	li	a0,1
1c009ebe:	0141                	addi	sp,sp,16
1c009ec0:	8082                	ret

1c009ec2 <__rt_pmu_init>:
1c009ec2:	1141                	addi	sp,sp,-16
1c009ec4:	1c0017b7          	lui	a5,0x1c001
1c009ec8:	1c00a5b7          	lui	a1,0x1c00a
1c009ecc:	46078793          	addi	a5,a5,1120 # 1c001460 <stack>
1c009ed0:	c606                	sw	ra,12(sp)
1c009ed2:	4741                	li	a4,16
1c009ed4:	f3458593          	addi	a1,a1,-204 # 1c009f34 <__rt_pmu_scu_handler>
1c009ed8:	4565                	li	a0,25
1c009eda:	c3d8                	sw	a4,4(a5)
1c009edc:	0007a023          	sw	zero,0(a5)
1c009ee0:	0007a423          	sw	zero,8(a5)
1c009ee4:	0007aa23          	sw	zero,20(a5)
1c009ee8:	2f5000ef          	jal	ra,1c00a9dc <rt_irq_set_handler>
1c009eec:	477d                	li	a4,31
1c009eee:	f14027f3          	csrr	a5,mhartid
1c009ef2:	ca5797b3          	p.extractu	a5,a5,5,5
1c009ef6:	02e79963          	bne	a5,a4,1c009f28 <__rt_pmu_init+0x66>
1c009efa:	1a1097b7          	lui	a5,0x1a109
1c009efe:	02000737          	lui	a4,0x2000
1c009f02:	c3d8                	sw	a4,4(a5)
1c009f04:	479d                	li	a5,7
1c009f06:	1a107737          	lui	a4,0x1a107
1c009f0a:	00f72623          	sw	a5,12(a4) # 1a10700c <__l1_end+0xa0ffff4>
1c009f0e:	40b2                	lw	ra,12(sp)
1c009f10:	00100737          	lui	a4,0x100
1c009f14:	02000793          	li	a5,32
1c009f18:	1741                	addi	a4,a4,-16
1c009f1a:	0007a023          	sw	zero,0(a5) # 1a109000 <__l1_end+0xa101fe8>
1c009f1e:	c798                	sw	a4,8(a5)
1c009f20:	0007a223          	sw	zero,4(a5)
1c009f24:	0141                	addi	sp,sp,16
1c009f26:	8082                	ret
1c009f28:	002047b7          	lui	a5,0x204
1c009f2c:	02000737          	lui	a4,0x2000
1c009f30:	cbd8                	sw	a4,20(a5)
1c009f32:	bfc9                	j	1c009f04 <__rt_pmu_init+0x42>

1c009f34 <__rt_pmu_scu_handler>:
1c009f34:	7179                	addi	sp,sp,-48
1c009f36:	cc3a                	sw	a4,24(sp)
1c009f38:	ca3e                	sw	a5,20(sp)
1c009f3a:	1a107737          	lui	a4,0x1a107
1c009f3e:	47c1                	li	a5,16
1c009f40:	d61a                	sw	t1,44(sp)
1c009f42:	d42a                	sw	a0,40(sp)
1c009f44:	d22e                	sw	a1,36(sp)
1c009f46:	d032                	sw	a2,32(sp)
1c009f48:	ce36                	sw	a3,28(sp)
1c009f4a:	c842                	sw	a6,16(sp)
1c009f4c:	c646                	sw	a7,12(sp)
1c009f4e:	00f72823          	sw	a5,16(a4) # 1a107010 <__l1_end+0xa0ffff8>
1c009f52:	02002783          	lw	a5,32(zero) # 20 <__rt_pmu_scu_event>
1c009f56:	1c001737          	lui	a4,0x1c001
1c009f5a:	46072023          	sw	zero,1120(a4) # 1c001460 <stack>
1c009f5e:	853a                	mv	a0,a4
1c009f60:	cf81                	beqz	a5,1c009f78 <__rt_pmu_scu_handler+0x44>
1c009f62:	00c02703          	lw	a4,12(zero) # c <__rt_sched>
1c009f66:	0007ac23          	sw	zero,24(a5) # 204018 <__l1_heap_size+0x1eb030>
1c009f6a:	e345                	bnez	a4,1c00a00a <__rt_pmu_scu_handler+0xd6>
1c009f6c:	00f02623          	sw	a5,12(zero) # c <__rt_sched>
1c009f70:	00f02823          	sw	a5,16(zero) # 10 <__rt_sched+0x4>
1c009f74:	02002023          	sw	zero,32(zero) # 20 <__rt_pmu_scu_event>
1c009f78:	1c0017b7          	lui	a5,0x1c001
1c009f7c:	4687a683          	lw	a3,1128(a5) # 1c001468 <__rt_pmu_pending_requests>
1c009f80:	caad                	beqz	a3,1c009ff2 <__rt_pmu_scu_handler+0xbe>
1c009f82:	4e98                	lw	a4,24(a3)
1c009f84:	1c0018b7          	lui	a7,0x1c001
1c009f88:	4648a303          	lw	t1,1124(a7) # 1c001464 <__rt_pmu_domains_on>
1c009f8c:	46e7a423          	sw	a4,1128(a5)
1c009f90:	5ad8                	lw	a4,52(a3)
1c009f92:	4785                	li	a5,1
1c009f94:	0386a803          	lw	a6,56(a3)
1c009f98:	00e79633          	sll	a2,a5,a4
1c009f9c:	fff64613          	not	a2,a2
1c009fa0:	00667633          	and	a2,a2,t1
1c009fa4:	00e81333          	sll	t1,a6,a4
1c009fa8:	00666633          	or	a2,a2,t1
1c009fac:	46c8a223          	sw	a2,1124(a7)
1c009fb0:	46f52023          	sw	a5,1120(a0)
1c009fb4:	5ecc                	lw	a1,60(a3)
1c009fb6:	04473e63          	p.bneimm	a4,4,1c00a012 <__rt_pmu_scu_handler+0xde>
1c009fba:	fc15b733          	p.bclr	a4,a1,30,1
1c009fbe:	8989                	andi	a1,a1,2
1c009fc0:	00470793          	addi	a5,a4,4
1c009fc4:	c199                	beqz	a1,1c009fca <__rt_pmu_scu_handler+0x96>
1c009fc6:	00e70793          	addi	a5,a4,14
1c009fca:	0037d713          	srli	a4,a5,0x3
1c009fce:	6641                	lui	a2,0x10
1c009fd0:	0709                	addi	a4,a4,2
1c009fd2:	f837b7b3          	p.bclr	a5,a5,28,3
1c009fd6:	0706                	slli	a4,a4,0x1
1c009fd8:	00f617b3          	sll	a5,a2,a5
1c009fdc:	8fd9                	or	a5,a5,a4
1c009fde:	0417e793          	ori	a5,a5,65
1c009fe2:	1a107737          	lui	a4,0x1a107
1c009fe6:	00f72023          	sw	a5,0(a4) # 1a107000 <__l1_end+0xa0fffe8>
1c009fea:	42bc                	lw	a5,64(a3)
1c009fec:	cb85                	beqz	a5,1c00a01c <__rt_pmu_scu_handler+0xe8>
1c009fee:	02d02023          	sw	a3,32(zero) # 20 <__rt_pmu_scu_event>
1c009ff2:	5332                	lw	t1,44(sp)
1c009ff4:	5522                	lw	a0,40(sp)
1c009ff6:	5592                	lw	a1,36(sp)
1c009ff8:	5602                	lw	a2,32(sp)
1c009ffa:	46f2                	lw	a3,28(sp)
1c009ffc:	4762                	lw	a4,24(sp)
1c009ffe:	47d2                	lw	a5,20(sp)
1c00a000:	4842                	lw	a6,16(sp)
1c00a002:	48b2                	lw	a7,12(sp)
1c00a004:	6145                	addi	sp,sp,48
1c00a006:	30200073          	mret
1c00a00a:	01002703          	lw	a4,16(zero) # 10 <__rt_sched+0x4>
1c00a00e:	cf1c                	sw	a5,24(a4)
1c00a010:	b785                	j	1c009f70 <__rt_pmu_scu_handler+0x3c>
1c00a012:	00370793          	addi	a5,a4,3
1c00a016:	0786                	slli	a5,a5,0x1
1c00a018:	97c2                	add	a5,a5,a6
1c00a01a:	bf45                	j	1c009fca <__rt_pmu_scu_handler+0x96>
1c00a01c:	00c02783          	lw	a5,12(zero) # c <__rt_sched>
1c00a020:	0006ac23          	sw	zero,24(a3)
1c00a024:	e791                	bnez	a5,1c00a030 <__rt_pmu_scu_handler+0xfc>
1c00a026:	00d02623          	sw	a3,12(zero) # c <__rt_sched>
1c00a02a:	00d02823          	sw	a3,16(zero) # 10 <__rt_sched+0x4>
1c00a02e:	b7d1                	j	1c009ff2 <__rt_pmu_scu_handler+0xbe>
1c00a030:	01002783          	lw	a5,16(zero) # 10 <__rt_sched+0x4>
1c00a034:	cf94                	sw	a3,24(a5)
1c00a036:	bfd5                	j	1c00a02a <__rt_pmu_scu_handler+0xf6>

1c00a038 <__rt_init_cluster_data>:
1c00a038:	04050713          	addi	a4,a0,64
1c00a03c:	00800793          	li	a5,8
1c00a040:	01671613          	slli	a2,a4,0x16
1c00a044:	e6c7b7b3          	p.bclr	a5,a5,19,12
1c00a048:	1c0106b7          	lui	a3,0x1c010
1c00a04c:	97b2                	add	a5,a5,a2
1c00a04e:	671d                	lui	a4,0x7
1c00a050:	19068693          	addi	a3,a3,400 # 1c010190 <_l1_preload_start_inL2>
1c00a054:	01070713          	addi	a4,a4,16 # 7010 <_l1_preload_size>
1c00a058:	8f95                	sub	a5,a5,a3
1c00a05a:	00f685b3          	add	a1,a3,a5
1c00a05e:	02e04963          	bgtz	a4,1c00a090 <__rt_init_cluster_data+0x58>
1c00a062:	1c0017b7          	lui	a5,0x1c001
1c00a066:	02800713          	li	a4,40
1c00a06a:	7b478793          	addi	a5,a5,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c00a06e:	42e507b3          	p.mac	a5,a0,a4
1c00a072:	00201737          	lui	a4,0x201
1c00a076:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e7e1c>
1c00a07a:	9732                	add	a4,a4,a2
1c00a07c:	cb98                	sw	a4,16(a5)
1c00a07e:	00800713          	li	a4,8
1c00a082:	e6c73733          	p.bclr	a4,a4,19,12
1c00a086:	9732                	add	a4,a4,a2
1c00a088:	0007a423          	sw	zero,8(a5)
1c00a08c:	cbd8                	sw	a4,20(a5)
1c00a08e:	8082                	ret
1c00a090:	0046a80b          	p.lw	a6,4(a3!)
1c00a094:	1771                	addi	a4,a4,-4
1c00a096:	0105a023          	sw	a6,0(a1)
1c00a09a:	b7c1                	j	1c00a05a <__rt_init_cluster_data+0x22>

1c00a09c <__rt_cluster_mount_step>:
1c00a09c:	7179                	addi	sp,sp,-48
1c00a09e:	ce4e                	sw	s3,28(sp)
1c00a0a0:	cc52                	sw	s4,24(sp)
1c00a0a2:	00800993          	li	s3,8
1c00a0a6:	1c008a37          	lui	s4,0x1c008
1c00a0aa:	d422                	sw	s0,40(sp)
1c00a0ac:	ca56                	sw	s5,20(sp)
1c00a0ae:	d606                	sw	ra,44(sp)
1c00a0b0:	d226                	sw	s1,36(sp)
1c00a0b2:	d04a                	sw	s2,32(sp)
1c00a0b4:	c85a                	sw	s6,16(sp)
1c00a0b6:	842a                	mv	s0,a0
1c00a0b8:	e6c9b9b3          	p.bclr	s3,s3,19,12
1c00a0bc:	1c001ab7          	lui	s5,0x1c001
1c00a0c0:	080a0a13          	addi	s4,s4,128 # 1c008080 <_start>
1c00a0c4:	4c5c                	lw	a5,28(s0)
1c00a0c6:	0217ad63          	p.beqimm	a5,1,1c00a100 <__rt_cluster_mount_step+0x64>
1c00a0ca:	0a27af63          	p.beqimm	a5,2,1c00a188 <__rt_cluster_mount_step+0xec>
1c00a0ce:	ebcd                	bnez	a5,1c00a180 <__rt_cluster_mount_step+0xe4>
1c00a0d0:	5018                	lw	a4,32(s0)
1c00a0d2:	00042c23          	sw	zero,24(s0)
1c00a0d6:	e719                	bnez	a4,1c00a0e4 <__rt_cluster_mount_step+0x48>
1c00a0d8:	5048                	lw	a0,36(s0)
1c00a0da:	006c                	addi	a1,sp,12
1c00a0dc:	c602                	sw	zero,12(sp)
1c00a0de:	33c1                	jal	1c009e9e <__rt_pmu_cluster_power_up>
1c00a0e0:	47b2                	lw	a5,12(sp)
1c00a0e2:	cc08                	sw	a0,24(s0)
1c00a0e4:	4c58                	lw	a4,28(s0)
1c00a0e6:	0705                	addi	a4,a4,1
1c00a0e8:	cc58                	sw	a4,28(s0)
1c00a0ea:	dfe9                	beqz	a5,1c00a0c4 <__rt_cluster_mount_step+0x28>
1c00a0ec:	50b2                	lw	ra,44(sp)
1c00a0ee:	5422                	lw	s0,40(sp)
1c00a0f0:	5492                	lw	s1,36(sp)
1c00a0f2:	5902                	lw	s2,32(sp)
1c00a0f4:	49f2                	lw	s3,28(sp)
1c00a0f6:	4a62                	lw	s4,24(sp)
1c00a0f8:	4ad2                	lw	s5,20(sp)
1c00a0fa:	4b42                	lw	s6,16(sp)
1c00a0fc:	6145                	addi	sp,sp,48
1c00a0fe:	8082                	ret
1c00a100:	02042b03          	lw	s6,32(s0)
1c00a104:	040b0493          	addi	s1,s6,64
1c00a108:	04da                	slli	s1,s1,0x16
1c00a10a:	009987b3          	add	a5,s3,s1
1c00a10e:	0007a223          	sw	zero,4(a5)
1c00a112:	0007a423          	sw	zero,8(a5)
1c00a116:	0007a023          	sw	zero,0(a5)
1c00a11a:	644aa783          	lw	a5,1604(s5) # 1c001644 <__rt_platform>
1c00a11e:	0017af63          	p.beqimm	a5,1,1c00a13c <__rt_cluster_mount_step+0xa0>
1c00a122:	4509                	li	a0,2
1c00a124:	793000ef          	jal	ra,1c00b0b6 <__rt_fll_init>
1c00a128:	1c0017b7          	lui	a5,0x1c001
1c00a12c:	7e878793          	addi	a5,a5,2024 # 1c0017e8 <__rt_freq_domains>
1c00a130:	478c                	lw	a1,8(a5)
1c00a132:	c9a9                	beqz	a1,1c00a184 <__rt_cluster_mount_step+0xe8>
1c00a134:	4601                	li	a2,0
1c00a136:	4509                	li	a0,2
1c00a138:	056010ef          	jal	ra,1c00b18e <rt_freq_set_and_get>
1c00a13c:	00200937          	lui	s2,0x200
1c00a140:	01248733          	add	a4,s1,s2
1c00a144:	4785                	li	a5,1
1c00a146:	02f72023          	sw	a5,32(a4)
1c00a14a:	855a                	mv	a0,s6
1c00a14c:	35f5                	jal	1c00a038 <__rt_init_cluster_data>
1c00a14e:	855a                	mv	a0,s6
1c00a150:	927ff0ef          	jal	ra,1c009a76 <__rt_alloc_init_l1>
1c00a154:	002017b7          	lui	a5,0x201
1c00a158:	40078793          	addi	a5,a5,1024 # 201400 <__l1_heap_size+0x1e8418>
1c00a15c:	577d                	li	a4,-1
1c00a15e:	04090913          	addi	s2,s2,64 # 200040 <__l1_heap_size+0x1e7058>
1c00a162:	00e4e7a3          	p.sw	a4,a5(s1)
1c00a166:	9926                	add	s2,s2,s1
1c00a168:	009250fb          	lp.setupi	x1,9,1c00a170 <__rt_cluster_mount_step+0xd4>
1c00a16c:	0149222b          	p.sw	s4,4(s2!)
1c00a170:	0001                	nop
1c00a172:	002007b7          	lui	a5,0x200
1c00a176:	07a1                	addi	a5,a5,8
1c00a178:	1ff00713          	li	a4,511
1c00a17c:	00e4e7a3          	p.sw	a4,a5(s1)
1c00a180:	4781                	li	a5,0
1c00a182:	b78d                	j	1c00a0e4 <__rt_cluster_mount_step+0x48>
1c00a184:	c788                	sw	a0,8(a5)
1c00a186:	bf5d                	j	1c00a13c <__rt_cluster_mount_step+0xa0>
1c00a188:	505c                	lw	a5,36(s0)
1c00a18a:	5b98                	lw	a4,48(a5)
1c00a18c:	d398                	sw	a4,32(a5)
1c00a18e:	5798                	lw	a4,40(a5)
1c00a190:	c398                	sw	a4,0(a5)
1c00a192:	57d8                	lw	a4,44(a5)
1c00a194:	c3d8                	sw	a4,4(a5)
1c00a196:	0207a823          	sw	zero,48(a5) # 200030 <__l1_heap_size+0x1e7048>
1c00a19a:	505c                	lw	a5,36(s0)
1c00a19c:	00c02703          	lw	a4,12(zero) # c <__rt_sched>
1c00a1a0:	0007ac23          	sw	zero,24(a5)
1c00a1a4:	cb01                	beqz	a4,1c00a1b4 <__rt_cluster_mount_step+0x118>
1c00a1a6:	01002703          	lw	a4,16(zero) # 10 <__rt_sched+0x4>
1c00a1aa:	cf1c                	sw	a5,24(a4)
1c00a1ac:	00f02823          	sw	a5,16(zero) # 10 <__rt_sched+0x4>
1c00a1b0:	4785                	li	a5,1
1c00a1b2:	bf0d                	j	1c00a0e4 <__rt_cluster_mount_step+0x48>
1c00a1b4:	00f02623          	sw	a5,12(zero) # c <__rt_sched>
1c00a1b8:	bfd5                	j	1c00a1ac <__rt_cluster_mount_step+0x110>

1c00a1ba <__rt_cluster_init>:
1c00a1ba:	1c001537          	lui	a0,0x1c001
1c00a1be:	1141                	addi	sp,sp,-16
1c00a1c0:	02800613          	li	a2,40
1c00a1c4:	4581                	li	a1,0
1c00a1c6:	7b450513          	addi	a0,a0,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c00a1ca:	c606                	sw	ra,12(sp)
1c00a1cc:	540010ef          	jal	ra,1c00b70c <memset>
1c00a1d0:	1c0095b7          	lui	a1,0x1c009
1c00a1d4:	ca658593          	addi	a1,a1,-858 # 1c008ca6 <__rt_remote_enqueue_event>
1c00a1d8:	4505                	li	a0,1
1c00a1da:	003000ef          	jal	ra,1c00a9dc <rt_irq_set_handler>
1c00a1de:	477d                	li	a4,31
1c00a1e0:	f14027f3          	csrr	a5,mhartid
1c00a1e4:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a1e8:	02e79c63          	bne	a5,a4,1c00a220 <__rt_cluster_init+0x66>
1c00a1ec:	1a1097b7          	lui	a5,0x1a109
1c00a1f0:	4709                	li	a4,2
1c00a1f2:	c3d8                	sw	a4,4(a5)
1c00a1f4:	1c0095b7          	lui	a1,0x1c009
1c00a1f8:	c6e58593          	addi	a1,a1,-914 # 1c008c6e <__rt_bridge_enqueue_event>
1c00a1fc:	4511                	li	a0,4
1c00a1fe:	7de000ef          	jal	ra,1c00a9dc <rt_irq_set_handler>
1c00a202:	477d                	li	a4,31
1c00a204:	f14027f3          	csrr	a5,mhartid
1c00a208:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a20c:	00e79f63          	bne	a5,a4,1c00a22a <__rt_cluster_init+0x70>
1c00a210:	1a1097b7          	lui	a5,0x1a109
1c00a214:	4741                	li	a4,16
1c00a216:	c3d8                	sw	a4,4(a5)
1c00a218:	40b2                	lw	ra,12(sp)
1c00a21a:	4501                	li	a0,0
1c00a21c:	0141                	addi	sp,sp,16
1c00a21e:	8082                	ret
1c00a220:	002047b7          	lui	a5,0x204
1c00a224:	4709                	li	a4,2
1c00a226:	cbd8                	sw	a4,20(a5)
1c00a228:	b7f1                	j	1c00a1f4 <__rt_cluster_init+0x3a>
1c00a22a:	002047b7          	lui	a5,0x204
1c00a22e:	4741                	li	a4,16
1c00a230:	cbd8                	sw	a4,20(a5)
1c00a232:	b7dd                	j	1c00a218 <__rt_cluster_init+0x5e>

1c00a234 <pi_cluster_conf_init>:
1c00a234:	00052223          	sw	zero,4(a0)
1c00a238:	8082                	ret

1c00a23a <pi_cluster_open>:
1c00a23a:	1101                	addi	sp,sp,-32
1c00a23c:	ce06                	sw	ra,28(sp)
1c00a23e:	cc22                	sw	s0,24(sp)
1c00a240:	ca26                	sw	s1,20(sp)
1c00a242:	c84a                	sw	s2,16(sp)
1c00a244:	c64e                	sw	s3,12(sp)
1c00a246:	30047973          	csrrci	s2,mstatus,8
1c00a24a:	00452983          	lw	s3,4(a0)
1c00a24e:	1c0014b7          	lui	s1,0x1c001
1c00a252:	02800793          	li	a5,40
1c00a256:	0049a703          	lw	a4,4(s3)
1c00a25a:	7b448493          	addi	s1,s1,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c00a25e:	42f704b3          	p.mac	s1,a4,a5
1c00a262:	c504                	sw	s1,8(a0)
1c00a264:	bf6ff0ef          	jal	ra,1c00965a <__rt_wait_event_prepare_blocking>
1c00a268:	477d                	li	a4,31
1c00a26a:	f14027f3          	csrr	a5,mhartid
1c00a26e:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a272:	842a                	mv	s0,a0
1c00a274:	04e79463          	bne	a5,a4,1c00a2bc <pi_cluster_open+0x82>
1c00a278:	511c                	lw	a5,32(a0)
1c00a27a:	0004ae23          	sw	zero,28(s1)
1c00a27e:	d0c8                	sw	a0,36(s1)
1c00a280:	d91c                	sw	a5,48(a0)
1c00a282:	411c                	lw	a5,0(a0)
1c00a284:	02052223          	sw	zero,36(a0)
1c00a288:	d51c                	sw	a5,40(a0)
1c00a28a:	415c                	lw	a5,4(a0)
1c00a28c:	c144                	sw	s1,4(a0)
1c00a28e:	d55c                	sw	a5,44(a0)
1c00a290:	1c00a7b7          	lui	a5,0x1c00a
1c00a294:	09c78793          	addi	a5,a5,156 # 1c00a09c <__rt_cluster_mount_step>
1c00a298:	c11c                	sw	a5,0(a0)
1c00a29a:	4785                	li	a5,1
1c00a29c:	d11c                	sw	a5,32(a0)
1c00a29e:	8526                	mv	a0,s1
1c00a2a0:	3bf5                	jal	1c00a09c <__rt_cluster_mount_step>
1c00a2a2:	8522                	mv	a0,s0
1c00a2a4:	d04ff0ef          	jal	ra,1c0097a8 <__rt_wait_event>
1c00a2a8:	30091073          	csrw	mstatus,s2
1c00a2ac:	40f2                	lw	ra,28(sp)
1c00a2ae:	4462                	lw	s0,24(sp)
1c00a2b0:	44d2                	lw	s1,20(sp)
1c00a2b2:	4942                	lw	s2,16(sp)
1c00a2b4:	49b2                	lw	s3,12(sp)
1c00a2b6:	4501                	li	a0,0
1c00a2b8:	6105                	addi	sp,sp,32
1c00a2ba:	8082                	ret
1c00a2bc:	0049a483          	lw	s1,4(s3)
1c00a2c0:	8526                	mv	a0,s1
1c00a2c2:	3b9d                	jal	1c00a038 <__rt_init_cluster_data>
1c00a2c4:	04048513          	addi	a0,s1,64
1c00a2c8:	002017b7          	lui	a5,0x201
1c00a2cc:	055a                	slli	a0,a0,0x16
1c00a2ce:	40078793          	addi	a5,a5,1024 # 201400 <__l1_heap_size+0x1e8418>
1c00a2d2:	577d                	li	a4,-1
1c00a2d4:	00e567a3          	p.sw	a4,a5(a0)
1c00a2d8:	002007b7          	lui	a5,0x200
1c00a2dc:	04478793          	addi	a5,a5,68 # 200044 <__l1_heap_size+0x1e705c>
1c00a2e0:	1c0086b7          	lui	a3,0x1c008
1c00a2e4:	97aa                	add	a5,a5,a0
1c00a2e6:	08068693          	addi	a3,a3,128 # 1c008080 <_start>
1c00a2ea:	008250fb          	lp.setupi	x1,8,1c00a2f2 <pi_cluster_open+0xb8>
1c00a2ee:	00d7a22b          	p.sw	a3,4(a5!)
1c00a2f2:	0001                	nop
1c00a2f4:	002007b7          	lui	a5,0x200
1c00a2f8:	07a1                	addi	a5,a5,8
1c00a2fa:	577d                	li	a4,-1
1c00a2fc:	00e567a3          	p.sw	a4,a5(a0)
1c00a300:	8522                	mv	a0,s0
1c00a302:	c22ff0ef          	jal	ra,1c009724 <rt_event_push>
1c00a306:	bf71                	j	1c00a2a2 <pi_cluster_open+0x68>

1c00a308 <pi_cluster_close>:
1c00a308:	451c                	lw	a5,8(a0)
1c00a30a:	1101                	addi	sp,sp,-32
1c00a30c:	cc22                	sw	s0,24(sp)
1c00a30e:	5380                	lw	s0,32(a5)
1c00a310:	1c0017b7          	lui	a5,0x1c001
1c00a314:	6447a783          	lw	a5,1604(a5) # 1c001644 <__rt_platform>
1c00a318:	ce06                	sw	ra,28(sp)
1c00a31a:	0017a563          	p.beqimm	a5,1,1c00a324 <pi_cluster_close+0x1c>
1c00a31e:	4509                	li	a0,2
1c00a320:	63b000ef          	jal	ra,1c00b15a <__rt_fll_deinit>
1c00a324:	c602                	sw	zero,12(sp)
1c00a326:	e401                	bnez	s0,1c00a32e <pi_cluster_close+0x26>
1c00a328:	006c                	addi	a1,sp,12
1c00a32a:	4501                	li	a0,0
1c00a32c:	36ad                	jal	1c009e96 <__rt_pmu_cluster_power_down>
1c00a32e:	40f2                	lw	ra,28(sp)
1c00a330:	4462                	lw	s0,24(sp)
1c00a332:	4501                	li	a0,0
1c00a334:	6105                	addi	sp,sp,32
1c00a336:	8082                	ret

1c00a338 <__rt_cluster_push_fc_event>:
1c00a338:	002047b7          	lui	a5,0x204
1c00a33c:	0c078793          	addi	a5,a5,192 # 2040c0 <__l1_heap_size+0x1eb0d8>
1c00a340:	0007e703          	p.elw	a4,0(a5)
1c00a344:	f1402773          	csrr	a4,mhartid
1c00a348:	1c0017b7          	lui	a5,0x1c001
1c00a34c:	8715                	srai	a4,a4,0x5
1c00a34e:	f2673733          	p.bclr	a4,a4,25,6
1c00a352:	02800693          	li	a3,40
1c00a356:	7b478793          	addi	a5,a5,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c00a35a:	42d707b3          	p.mac	a5,a4,a3
1c00a35e:	4689                	li	a3,2
1c00a360:	00204737          	lui	a4,0x204
1c00a364:	43d0                	lw	a2,4(a5)
1c00a366:	ea19                	bnez	a2,1c00a37c <__rt_cluster_push_fc_event+0x44>
1c00a368:	c3c8                	sw	a0,4(a5)
1c00a36a:	4709                	li	a4,2
1c00a36c:	1a1097b7          	lui	a5,0x1a109
1c00a370:	cb98                	sw	a4,16(a5)
1c00a372:	002047b7          	lui	a5,0x204
1c00a376:	0c07a023          	sw	zero,192(a5) # 2040c0 <__l1_heap_size+0x1eb0d8>
1c00a37a:	8082                	ret
1c00a37c:	c714                	sw	a3,8(a4)
1c00a37e:	03c76603          	p.elw	a2,60(a4) # 20403c <__l1_heap_size+0x1eb054>
1c00a382:	c354                	sw	a3,4(a4)
1c00a384:	b7c5                	j	1c00a364 <__rt_cluster_push_fc_event+0x2c>

1c00a386 <__rt_cluster_new>:
1c00a386:	1c00a5b7          	lui	a1,0x1c00a
1c00a38a:	1141                	addi	sp,sp,-16
1c00a38c:	4601                	li	a2,0
1c00a38e:	1ba58593          	addi	a1,a1,442 # 1c00a1ba <__rt_cluster_init>
1c00a392:	4501                	li	a0,0
1c00a394:	c606                	sw	ra,12(sp)
1c00a396:	7d4000ef          	jal	ra,1c00ab6a <__rt_cbsys_add>
1c00a39a:	c10d                	beqz	a0,1c00a3bc <__rt_cluster_new+0x36>
1c00a39c:	f1402673          	csrr	a2,mhartid
1c00a3a0:	1c001537          	lui	a0,0x1c001
1c00a3a4:	40565593          	srai	a1,a2,0x5
1c00a3a8:	f265b5b3          	p.bclr	a1,a1,25,6
1c00a3ac:	f4563633          	p.bclr	a2,a2,26,5
1c00a3b0:	a3450513          	addi	a0,a0,-1484 # 1c000a34 <PIo2+0xd8>
1c00a3b4:	5b2010ef          	jal	ra,1c00b966 <printf>
1c00a3b8:	53c010ef          	jal	ra,1c00b8f4 <abort>
1c00a3bc:	40b2                	lw	ra,12(sp)
1c00a3be:	0141                	addi	sp,sp,16
1c00a3c0:	8082                	ret

1c00a3c2 <__rt_cluster_pulpos_emu_init>:
1c00a3c2:	1141                	addi	sp,sp,-16
1c00a3c4:	45b1                	li	a1,12
1c00a3c6:	4501                	li	a0,0
1c00a3c8:	c606                	sw	ra,12(sp)
1c00a3ca:	e3eff0ef          	jal	ra,1c009a08 <rt_alloc>
1c00a3ce:	1c0017b7          	lui	a5,0x1c001
1c00a3d2:	74a7a023          	sw	a0,1856(a5) # 1c001740 <__rt_fc_cluster_device>
1c00a3d6:	e10d                	bnez	a0,1c00a3f8 <__rt_cluster_pulpos_emu_init+0x36>
1c00a3d8:	f1402673          	csrr	a2,mhartid
1c00a3dc:	1c001537          	lui	a0,0x1c001
1c00a3e0:	40565593          	srai	a1,a2,0x5
1c00a3e4:	f265b5b3          	p.bclr	a1,a1,25,6
1c00a3e8:	f4563633          	p.bclr	a2,a2,26,5
1c00a3ec:	a7c50513          	addi	a0,a0,-1412 # 1c000a7c <PIo2+0x120>
1c00a3f0:	576010ef          	jal	ra,1c00b966 <printf>
1c00a3f4:	500010ef          	jal	ra,1c00b8f4 <abort>
1c00a3f8:	40b2                	lw	ra,12(sp)
1c00a3fa:	0141                	addi	sp,sp,16
1c00a3fc:	8082                	ret

1c00a3fe <rt_cluster_call>:
1c00a3fe:	7139                	addi	sp,sp,-64
1c00a400:	d84a                	sw	s2,48(sp)
1c00a402:	4906                	lw	s2,64(sp)
1c00a404:	dc22                	sw	s0,56(sp)
1c00a406:	842e                	mv	s0,a1
1c00a408:	de06                	sw	ra,60(sp)
1c00a40a:	da26                	sw	s1,52(sp)
1c00a40c:	d64e                	sw	s3,44(sp)
1c00a40e:	300479f3          	csrrci	s3,mstatus,8
1c00a412:	84ca                	mv	s1,s2
1c00a414:	02091163          	bnez	s2,1c00a436 <rt_cluster_call+0x38>
1c00a418:	ce32                	sw	a2,28(sp)
1c00a41a:	cc36                	sw	a3,24(sp)
1c00a41c:	ca3a                	sw	a4,20(sp)
1c00a41e:	c83e                	sw	a5,16(sp)
1c00a420:	c642                	sw	a6,12(sp)
1c00a422:	c446                	sw	a7,8(sp)
1c00a424:	a36ff0ef          	jal	ra,1c00965a <__rt_wait_event_prepare_blocking>
1c00a428:	48a2                	lw	a7,8(sp)
1c00a42a:	4832                	lw	a6,12(sp)
1c00a42c:	47c2                	lw	a5,16(sp)
1c00a42e:	4752                	lw	a4,20(sp)
1c00a430:	46e2                	lw	a3,24(sp)
1c00a432:	4672                	lw	a2,28(sp)
1c00a434:	84aa                	mv	s1,a0
1c00a436:	1c0015b7          	lui	a1,0x1c001
1c00a43a:	65058513          	addi	a0,a1,1616 # 1c001650 <_edata>
1c00a43e:	c55c                	sw	a5,12(a0)
1c00a440:	1c0017b7          	lui	a5,0x1c001
1c00a444:	c110                	sw	a2,0(a0)
1c00a446:	c154                	sw	a3,4(a0)
1c00a448:	c518                	sw	a4,8(a0)
1c00a44a:	01052823          	sw	a6,16(a0)
1c00a44e:	01152a23          	sw	a7,20(a0)
1c00a452:	7407a503          	lw	a0,1856(a5) # 1c001740 <__rt_fc_cluster_device>
1c00a456:	47b1                	li	a5,12
1c00a458:	8626                	mv	a2,s1
1c00a45a:	42f40533          	p.mac	a0,s0,a5
1c00a45e:	65058593          	addi	a1,a1,1616
1c00a462:	2041                	jal	1c00a4e2 <pi_cluster_send_task_to_cl_async>
1c00a464:	842a                	mv	s0,a0
1c00a466:	cd01                	beqz	a0,1c00a47e <rt_cluster_call+0x80>
1c00a468:	30099073          	csrw	mstatus,s3
1c00a46c:	547d                	li	s0,-1
1c00a46e:	8522                	mv	a0,s0
1c00a470:	50f2                	lw	ra,60(sp)
1c00a472:	5462                	lw	s0,56(sp)
1c00a474:	54d2                	lw	s1,52(sp)
1c00a476:	5942                	lw	s2,48(sp)
1c00a478:	59b2                	lw	s3,44(sp)
1c00a47a:	6121                	addi	sp,sp,64
1c00a47c:	8082                	ret
1c00a47e:	00091563          	bnez	s2,1c00a488 <rt_cluster_call+0x8a>
1c00a482:	8526                	mv	a0,s1
1c00a484:	b24ff0ef          	jal	ra,1c0097a8 <__rt_wait_event>
1c00a488:	30099073          	csrw	mstatus,s3
1c00a48c:	b7cd                	j	1c00a46e <rt_cluster_call+0x70>

1c00a48e <rt_cluster_mount>:
1c00a48e:	7139                	addi	sp,sp,-64
1c00a490:	dc22                	sw	s0,56(sp)
1c00a492:	da26                	sw	s1,52(sp)
1c00a494:	d84a                	sw	s2,48(sp)
1c00a496:	4431                	li	s0,12
1c00a498:	1c0014b7          	lui	s1,0x1c001
1c00a49c:	de06                	sw	ra,60(sp)
1c00a49e:	d64e                	sw	s3,44(sp)
1c00a4a0:	8936                	mv	s2,a3
1c00a4a2:	02858433          	mul	s0,a1,s0
1c00a4a6:	74048493          	addi	s1,s1,1856 # 1c001740 <__rt_fc_cluster_device>
1c00a4aa:	c905                	beqz	a0,1c00a4da <rt_cluster_mount+0x4c>
1c00a4ac:	0068                	addi	a0,sp,12
1c00a4ae:	89ae                	mv	s3,a1
1c00a4b0:	3351                	jal	1c00a234 <pi_cluster_conf_init>
1c00a4b2:	4088                	lw	a0,0(s1)
1c00a4b4:	006c                	addi	a1,sp,12
1c00a4b6:	9522                	add	a0,a0,s0
1c00a4b8:	2305                	jal	1c00a9d8 <pi_open_from_conf>
1c00a4ba:	4088                	lw	a0,0(s1)
1c00a4bc:	c84e                	sw	s3,16(sp)
1c00a4be:	9522                	add	a0,a0,s0
1c00a4c0:	3bad                	jal	1c00a23a <pi_cluster_open>
1c00a4c2:	00090563          	beqz	s2,1c00a4cc <rt_cluster_mount+0x3e>
1c00a4c6:	854a                	mv	a0,s2
1c00a4c8:	a5cff0ef          	jal	ra,1c009724 <rt_event_push>
1c00a4cc:	50f2                	lw	ra,60(sp)
1c00a4ce:	5462                	lw	s0,56(sp)
1c00a4d0:	54d2                	lw	s1,52(sp)
1c00a4d2:	5942                	lw	s2,48(sp)
1c00a4d4:	59b2                	lw	s3,44(sp)
1c00a4d6:	6121                	addi	sp,sp,64
1c00a4d8:	8082                	ret
1c00a4da:	4088                	lw	a0,0(s1)
1c00a4dc:	9522                	add	a0,a0,s0
1c00a4de:	352d                	jal	1c00a308 <pi_cluster_close>
1c00a4e0:	b7cd                	j	1c00a4c2 <rt_cluster_mount+0x34>

1c00a4e2 <pi_cluster_send_task_to_cl_async>:
1c00a4e2:	1101                	addi	sp,sp,-32
1c00a4e4:	ca26                	sw	s1,20(sp)
1c00a4e6:	4504                	lw	s1,8(a0)
1c00a4e8:	cc22                	sw	s0,24(sp)
1c00a4ea:	c256                	sw	s5,4(sp)
1c00a4ec:	842e                	mv	s0,a1
1c00a4ee:	8ab2                	mv	s5,a2
1c00a4f0:	ce06                	sw	ra,28(sp)
1c00a4f2:	c84a                	sw	s2,16(sp)
1c00a4f4:	c64e                	sw	s3,12(sp)
1c00a4f6:	c452                	sw	s4,8(sp)
1c00a4f8:	30047a73          	csrrci	s4,mstatus,8
1c00a4fc:	00060823          	sb	zero,16(a2) # 10010 <_l1_preload_size+0x9000>
1c00a500:	4785                	li	a5,1
1c00a502:	d1dc                	sw	a5,36(a1)
1c00a504:	49dc                	lw	a5,20(a1)
1c00a506:	0144a983          	lw	s3,20(s1)
1c00a50a:	e399                	bnez	a5,1c00a510 <pi_cluster_send_task_to_cl_async+0x2e>
1c00a50c:	47a5                	li	a5,9
1c00a50e:	c9dc                	sw	a5,20(a1)
1c00a510:	441c                	lw	a5,8(s0)
1c00a512:	ef85                	bnez	a5,1c00a54a <pi_cluster_send_task_to_cl_async+0x68>
1c00a514:	445c                	lw	a5,12(s0)
1c00a516:	eb81                	bnez	a5,1c00a526 <pi_cluster_send_task_to_cl_async+0x44>
1c00a518:	6785                	lui	a5,0x1
1c00a51a:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c00a51e:	c45c                	sw	a5,12(s0)
1c00a520:	40000793          	li	a5,1024
1c00a524:	c81c                	sw	a5,16(s0)
1c00a526:	481c                	lw	a5,16(s0)
1c00a528:	00c42903          	lw	s2,12(s0)
1c00a52c:	e399                	bnez	a5,1c00a532 <pi_cluster_send_task_to_cl_async+0x50>
1c00a52e:	01242823          	sw	s2,16(s0)
1c00a532:	485c                	lw	a5,20(s0)
1c00a534:	4818                	lw	a4,16(s0)
1c00a536:	448c                	lw	a1,8(s1)
1c00a538:	17fd                	addi	a5,a5,-1
1c00a53a:	42e78933          	p.mac	s2,a5,a4
1c00a53e:	cdad                	beqz	a1,1c00a5b8 <pi_cluster_send_task_to_cl_async+0xd6>
1c00a540:	44d0                	lw	a2,12(s1)
1c00a542:	07261163          	bne	a2,s2,1c00a5a4 <pi_cluster_send_task_to_cl_async+0xc2>
1c00a546:	449c                	lw	a5,8(s1)
1c00a548:	c41c                	sw	a5,8(s0)
1c00a54a:	485c                	lw	a5,20(s0)
1c00a54c:	01542c23          	sw	s5,24(s0)
1c00a550:	02042023          	sw	zero,32(s0)
1c00a554:	fff78713          	addi	a4,a5,-1
1c00a558:	4785                	li	a5,1
1c00a55a:	00e797b3          	sll	a5,a5,a4
1c00a55e:	17fd                	addi	a5,a5,-1
1c00a560:	d41c                	sw	a5,40(s0)
1c00a562:	0089a783          	lw	a5,8(s3)
1c00a566:	cfa5                	beqz	a5,1c00a5de <pi_cluster_send_task_to_cl_async+0xfc>
1c00a568:	d380                	sw	s0,32(a5)
1c00a56a:	0089a423          	sw	s0,8(s3)
1c00a56e:	0009a783          	lw	a5,0(s3)
1c00a572:	e399                	bnez	a5,1c00a578 <pi_cluster_send_task_to_cl_async+0x96>
1c00a574:	0089a023          	sw	s0,0(s3)
1c00a578:	509c                	lw	a5,32(s1)
1c00a57a:	00201737          	lui	a4,0x201
1c00a57e:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e7e1c>
1c00a582:	04078793          	addi	a5,a5,64
1c00a586:	07da                	slli	a5,a5,0x16
1c00a588:	0007e723          	p.sw	zero,a4(a5)
1c00a58c:	300a1073          	csrw	mstatus,s4
1c00a590:	4501                	li	a0,0
1c00a592:	40f2                	lw	ra,28(sp)
1c00a594:	4462                	lw	s0,24(sp)
1c00a596:	44d2                	lw	s1,20(sp)
1c00a598:	4942                	lw	s2,16(sp)
1c00a59a:	49b2                	lw	s3,12(sp)
1c00a59c:	4a22                	lw	s4,8(sp)
1c00a59e:	4a92                	lw	s5,4(sp)
1c00a5a0:	6105                	addi	sp,sp,32
1c00a5a2:	8082                	ret
1c00a5a4:	1c001737          	lui	a4,0x1c001
1c00a5a8:	76072503          	lw	a0,1888(a4) # 1c001760 <__rt_alloc_l1>
1c00a5ac:	509c                	lw	a5,32(s1)
1c00a5ae:	4761                	li	a4,24
1c00a5b0:	42f70533          	p.mac	a0,a4,a5
1c00a5b4:	bcaff0ef          	jal	ra,1c00997e <rt_user_free>
1c00a5b8:	1c001737          	lui	a4,0x1c001
1c00a5bc:	76072503          	lw	a0,1888(a4) # 1c001760 <__rt_alloc_l1>
1c00a5c0:	509c                	lw	a5,32(s1)
1c00a5c2:	4761                	li	a4,24
1c00a5c4:	0124a623          	sw	s2,12(s1)
1c00a5c8:	42f70533          	p.mac	a0,a4,a5
1c00a5cc:	85ca                	mv	a1,s2
1c00a5ce:	b3eff0ef          	jal	ra,1c00990c <rt_user_alloc>
1c00a5d2:	c488                	sw	a0,8(s1)
1c00a5d4:	f92d                	bnez	a0,1c00a546 <pi_cluster_send_task_to_cl_async+0x64>
1c00a5d6:	300a1073          	csrw	mstatus,s4
1c00a5da:	557d                	li	a0,-1
1c00a5dc:	bf5d                	j	1c00a592 <pi_cluster_send_task_to_cl_async+0xb0>
1c00a5de:	0089a223          	sw	s0,4(s3)
1c00a5e2:	b761                	j	1c00a56a <pi_cluster_send_task_to_cl_async+0x88>

1c00a5e4 <cpu_perf_get>:
1c00a5e4:	10e52763          	p.beqimm	a0,14,1c00a6f2 <cpu_perf_get+0x10e>
1c00a5e8:	47b9                	li	a5,14
1c00a5ea:	04a7ee63          	bltu	a5,a0,1c00a646 <cpu_perf_get+0x62>
1c00a5ee:	0e652063          	p.beqimm	a0,6,1c00a6ce <cpu_perf_get+0xea>
1c00a5f2:	4799                	li	a5,6
1c00a5f4:	02a7e463          	bltu	a5,a0,1c00a61c <cpu_perf_get+0x38>
1c00a5f8:	0c252263          	p.beqimm	a0,2,1c00a6bc <cpu_perf_get+0xd8>
1c00a5fc:	4789                	li	a5,2
1c00a5fe:	00a7e763          	bltu	a5,a0,1c00a60c <cpu_perf_get+0x28>
1c00a602:	c55d                	beqz	a0,1c00a6b0 <cpu_perf_get+0xcc>
1c00a604:	0a152963          	p.beqimm	a0,1,1c00a6b6 <cpu_perf_get+0xd2>
1c00a608:	4501                	li	a0,0
1c00a60a:	8082                	ret
1c00a60c:	0a452e63          	p.beqimm	a0,4,1c00a6c8 <cpu_perf_get+0xe4>
1c00a610:	4791                	li	a5,4
1c00a612:	0aa7f863          	bleu	a0,a5,1c00a6c2 <cpu_perf_get+0xde>
1c00a616:	78502573          	csrr	a0,pccr5
1c00a61a:	8082                	ret
1c00a61c:	0ca52263          	p.beqimm	a0,10,1c00a6e0 <cpu_perf_get+0xfc>
1c00a620:	47a9                	li	a5,10
1c00a622:	00a7ea63          	bltu	a5,a0,1c00a636 <cpu_perf_get+0x52>
1c00a626:	0a852a63          	p.beqimm	a0,8,1c00a6da <cpu_perf_get+0xf6>
1c00a62a:	47a1                	li	a5,8
1c00a62c:	0aa7f463          	bleu	a0,a5,1c00a6d4 <cpu_perf_get+0xf0>
1c00a630:	78902573          	csrr	a0,pccr9
1c00a634:	8082                	ret
1c00a636:	0ac52b63          	p.beqimm	a0,12,1c00a6ec <cpu_perf_get+0x108>
1c00a63a:	47b1                	li	a5,12
1c00a63c:	0aa7f563          	bleu	a0,a5,1c00a6e6 <cpu_perf_get+0x102>
1c00a640:	78d02573          	csrr	a0,pccr13
1c00a644:	8082                	ret
1c00a646:	47dd                	li	a5,23
1c00a648:	0cf50763          	beq	a0,a5,1c00a716 <cpu_perf_get+0x132>
1c00a64c:	02a7ea63          	bltu	a5,a0,1c00a680 <cpu_perf_get+0x9c>
1c00a650:	47cd                	li	a5,19
1c00a652:	0af50963          	beq	a0,a5,1c00a704 <cpu_perf_get+0x120>
1c00a656:	00a7ed63          	bltu	a5,a0,1c00a670 <cpu_perf_get+0x8c>
1c00a65a:	47c1                	li	a5,16
1c00a65c:	0af50163          	beq	a0,a5,1c00a6fe <cpu_perf_get+0x11a>
1c00a660:	08f56c63          	bltu	a0,a5,1c00a6f8 <cpu_perf_get+0x114>
1c00a664:	47c9                	li	a5,18
1c00a666:	faf511e3          	bne	a0,a5,1c00a608 <cpu_perf_get+0x24>
1c00a66a:	79202573          	csrr	a0,pccr18
1c00a66e:	8082                	ret
1c00a670:	47d5                	li	a5,21
1c00a672:	08f50f63          	beq	a0,a5,1c00a710 <cpu_perf_get+0x12c>
1c00a676:	08a7fa63          	bleu	a0,a5,1c00a70a <cpu_perf_get+0x126>
1c00a67a:	79602573          	csrr	a0,pccr22
1c00a67e:	8082                	ret
1c00a680:	47ed                	li	a5,27
1c00a682:	0af50363          	beq	a0,a5,1c00a728 <cpu_perf_get+0x144>
1c00a686:	00a7ea63          	bltu	a5,a0,1c00a69a <cpu_perf_get+0xb6>
1c00a68a:	47e5                	li	a5,25
1c00a68c:	08f50b63          	beq	a0,a5,1c00a722 <cpu_perf_get+0x13e>
1c00a690:	08a7f663          	bleu	a0,a5,1c00a71c <cpu_perf_get+0x138>
1c00a694:	79a02573          	csrr	a0,pccr26
1c00a698:	8082                	ret
1c00a69a:	47f5                	li	a5,29
1c00a69c:	08f50c63          	beq	a0,a5,1c00a734 <cpu_perf_get+0x150>
1c00a6a0:	08f56763          	bltu	a0,a5,1c00a72e <cpu_perf_get+0x14a>
1c00a6a4:	47f9                	li	a5,30
1c00a6a6:	f6f511e3          	bne	a0,a5,1c00a608 <cpu_perf_get+0x24>
1c00a6aa:	79e02573          	csrr	a0,pccr30
1c00a6ae:	8082                	ret
1c00a6b0:	78002573          	csrr	a0,pccr0
1c00a6b4:	8082                	ret
1c00a6b6:	78102573          	csrr	a0,pccr1
1c00a6ba:	8082                	ret
1c00a6bc:	78202573          	csrr	a0,pccr2
1c00a6c0:	8082                	ret
1c00a6c2:	78302573          	csrr	a0,pccr3
1c00a6c6:	8082                	ret
1c00a6c8:	78402573          	csrr	a0,pccr4
1c00a6cc:	8082                	ret
1c00a6ce:	78602573          	csrr	a0,pccr6
1c00a6d2:	8082                	ret
1c00a6d4:	78702573          	csrr	a0,pccr7
1c00a6d8:	8082                	ret
1c00a6da:	78802573          	csrr	a0,pccr8
1c00a6de:	8082                	ret
1c00a6e0:	78a02573          	csrr	a0,pccr10
1c00a6e4:	8082                	ret
1c00a6e6:	78b02573          	csrr	a0,pccr11
1c00a6ea:	8082                	ret
1c00a6ec:	78c02573          	csrr	a0,pccr12
1c00a6f0:	8082                	ret
1c00a6f2:	78e02573          	csrr	a0,pccr14
1c00a6f6:	8082                	ret
1c00a6f8:	78f02573          	csrr	a0,pccr15
1c00a6fc:	8082                	ret
1c00a6fe:	79002573          	csrr	a0,pccr16
1c00a702:	8082                	ret
1c00a704:	79302573          	csrr	a0,pccr19
1c00a708:	8082                	ret
1c00a70a:	79402573          	csrr	a0,pccr20
1c00a70e:	8082                	ret
1c00a710:	79502573          	csrr	a0,pccr21
1c00a714:	8082                	ret
1c00a716:	79702573          	csrr	a0,pccr23
1c00a71a:	8082                	ret
1c00a71c:	79802573          	csrr	a0,pccr24
1c00a720:	8082                	ret
1c00a722:	79902573          	csrr	a0,pccr25
1c00a726:	8082                	ret
1c00a728:	79b02573          	csrr	a0,pccr27
1c00a72c:	8082                	ret
1c00a72e:	79c02573          	csrr	a0,pccr28
1c00a732:	8082                	ret
1c00a734:	79d02573          	csrr	a0,pccr29
1c00a738:	8082                	ret

1c00a73a <rt_perf_init>:
1c00a73a:	0511                	addi	a0,a0,4
1c00a73c:	012250fb          	lp.setupi	x1,18,1c00a744 <rt_perf_init+0xa>
1c00a740:	0005222b          	p.sw	zero,4(a0!)
1c00a744:	0001                	nop
1c00a746:	8082                	ret

1c00a748 <rt_perf_conf>:
1c00a748:	c10c                	sw	a1,0(a0)
1c00a74a:	cc059073          	csrw	0xcc0,a1
1c00a74e:	8082                	ret

1c00a750 <rt_perf_save>:
1c00a750:	7179                	addi	sp,sp,-48
1c00a752:	d04a                	sw	s2,32(sp)
1c00a754:	00052903          	lw	s2,0(a0)
1c00a758:	d226                	sw	s1,36(sp)
1c00a75a:	ce4e                	sw	s3,28(sp)
1c00a75c:	f14024f3          	csrr	s1,mhartid
1c00a760:	102009b7          	lui	s3,0x10200
1c00a764:	8495                	srai	s1,s1,0x5
1c00a766:	cc52                	sw	s4,24(sp)
1c00a768:	ca56                	sw	s5,20(sp)
1c00a76a:	c85a                	sw	s6,16(sp)
1c00a76c:	c65e                	sw	s7,12(sp)
1c00a76e:	d606                	sw	ra,44(sp)
1c00a770:	d422                	sw	s0,40(sp)
1c00a772:	8baa                	mv	s7,a0
1c00a774:	4a85                	li	s5,1
1c00a776:	f264b4b3          	p.bclr	s1,s1,25,6
1c00a77a:	4b7d                	li	s6,31
1c00a77c:	4a45                	li	s4,17
1c00a77e:	40098993          	addi	s3,s3,1024 # 10200400 <__l1_end+0x1f93e8>
1c00a782:	00091d63          	bnez	s2,1c00a79c <rt_perf_save+0x4c>
1c00a786:	50b2                	lw	ra,44(sp)
1c00a788:	5422                	lw	s0,40(sp)
1c00a78a:	5492                	lw	s1,36(sp)
1c00a78c:	5902                	lw	s2,32(sp)
1c00a78e:	49f2                	lw	s3,28(sp)
1c00a790:	4a62                	lw	s4,24(sp)
1c00a792:	4ad2                	lw	s5,20(sp)
1c00a794:	4b42                	lw	s6,16(sp)
1c00a796:	4bb2                	lw	s7,12(sp)
1c00a798:	6145                	addi	sp,sp,48
1c00a79a:	8082                	ret
1c00a79c:	10091533          	p.fl1	a0,s2
1c00a7a0:	00aa97b3          	sll	a5,s5,a0
1c00a7a4:	fff7c793          	not	a5,a5
1c00a7a8:	00f97933          	and	s2,s2,a5
1c00a7ac:	00251413          	slli	s0,a0,0x2
1c00a7b0:	01649d63          	bne	s1,s6,1c00a7ca <rt_perf_save+0x7a>
1c00a7b4:	03451063          	bne	a0,s4,1c00a7d4 <rt_perf_save+0x84>
1c00a7b8:	1a10b537          	lui	a0,0x1a10b
1c00a7bc:	00852503          	lw	a0,8(a0) # 1a10b008 <__l1_end+0xa103ff0>
1c00a7c0:	945e                	add	s0,s0,s7
1c00a7c2:	405c                	lw	a5,4(s0)
1c00a7c4:	953e                	add	a0,a0,a5
1c00a7c6:	c048                	sw	a0,4(s0)
1c00a7c8:	bf6d                	j	1c00a782 <rt_perf_save+0x32>
1c00a7ca:	01451563          	bne	a0,s4,1c00a7d4 <rt_perf_save+0x84>
1c00a7ce:	0089a503          	lw	a0,8(s3)
1c00a7d2:	b7fd                	j	1c00a7c0 <rt_perf_save+0x70>
1c00a7d4:	3d01                	jal	1c00a5e4 <cpu_perf_get>
1c00a7d6:	b7ed                	j	1c00a7c0 <rt_perf_save+0x70>

1c00a7d8 <cluster_start>:
1c00a7d8:	002047b7          	lui	a5,0x204
1c00a7dc:	00070737          	lui	a4,0x70
1c00a7e0:	c798                	sw	a4,8(a5)
1c00a7e2:	1ff00713          	li	a4,511
1c00a7e6:	002046b7          	lui	a3,0x204
1c00a7ea:	08e6a223          	sw	a4,132(a3) # 204084 <__l1_heap_size+0x1eb09c>
1c00a7ee:	20078693          	addi	a3,a5,512 # 204200 <__l1_heap_size+0x1eb218>
1c00a7f2:	c298                	sw	a4,0(a3)
1c00a7f4:	20c78793          	addi	a5,a5,524
1c00a7f8:	c398                	sw	a4,0(a5)
1c00a7fa:	8082                	ret

1c00a7fc <__rt_init>:
1c00a7fc:	1101                	addi	sp,sp,-32
1c00a7fe:	ce06                	sw	ra,28(sp)
1c00a800:	cc22                	sw	s0,24(sp)
1c00a802:	2d19                	jal	1c00ae18 <__rt_bridge_set_available>
1c00a804:	1c0017b7          	lui	a5,0x1c001
1c00a808:	6447a783          	lw	a5,1604(a5) # 1c001644 <__rt_platform>
1c00a80c:	0237b263          	p.bneimm	a5,3,1c00a830 <__rt_init+0x34>
1c00a810:	7d005073          	csrwi	0x7d0,0
1c00a814:	1c0017b7          	lui	a5,0x1c001
1c00a818:	c6078793          	addi	a5,a5,-928 # 1c000c60 <stack_start>
1c00a81c:	7d179073          	csrw	0x7d1,a5
1c00a820:	1c0017b7          	lui	a5,0x1c001
1c00a824:	46078793          	addi	a5,a5,1120 # 1c001460 <stack>
1c00a828:	7d279073          	csrw	0x7d2,a5
1c00a82c:	7d00d073          	csrwi	0x7d0,1
1c00a830:	24b1                	jal	1c00aa7c <__rt_irq_init>
1c00a832:	1a1067b7          	lui	a5,0x1a106
1c00a836:	577d                	li	a4,-1
1c00a838:	00478693          	addi	a3,a5,4 # 1a106004 <__l1_end+0xa0fefec>
1c00a83c:	c298                	sw	a4,0(a3)
1c00a83e:	00878693          	addi	a3,a5,8
1c00a842:	c298                	sw	a4,0(a3)
1c00a844:	00c78693          	addi	a3,a5,12
1c00a848:	c298                	sw	a4,0(a3)
1c00a84a:	01078693          	addi	a3,a5,16
1c00a84e:	c298                	sw	a4,0(a3)
1c00a850:	01478693          	addi	a3,a5,20
1c00a854:	c298                	sw	a4,0(a3)
1c00a856:	01878693          	addi	a3,a5,24
1c00a85a:	c298                	sw	a4,0(a3)
1c00a85c:	01c78693          	addi	a3,a5,28
1c00a860:	c298                	sw	a4,0(a3)
1c00a862:	02078793          	addi	a5,a5,32
1c00a866:	1c0095b7          	lui	a1,0x1c009
1c00a86a:	c398                	sw	a4,0(a5)
1c00a86c:	e5658593          	addi	a1,a1,-426 # 1c008e56 <__rt_fc_socevents_handler>
1c00a870:	4569                	li	a0,26
1c00a872:	22ad                	jal	1c00a9dc <rt_irq_set_handler>
1c00a874:	477d                	li	a4,31
1c00a876:	f14027f3          	csrr	a5,mhartid
1c00a87a:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a87e:	0ce79163          	bne	a5,a4,1c00a940 <__rt_init+0x144>
1c00a882:	1a1097b7          	lui	a5,0x1a109
1c00a886:	04000737          	lui	a4,0x4000
1c00a88a:	c3d8                	sw	a4,4(a5)
1c00a88c:	e36ff0ef          	jal	ra,1c009ec2 <__rt_pmu_init>
1c00a890:	13b000ef          	jal	ra,1c00b1ca <__rt_freq_init>
1c00a894:	477d                	li	a4,31
1c00a896:	f14027f3          	csrr	a5,mhartid
1c00a89a:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a89e:	0ae79763          	bne	a5,a4,1c00a94c <__rt_init+0x150>
1c00a8a2:	1a1097b7          	lui	a5,0x1a109
1c00a8a6:	577d                	li	a4,-1
1c00a8a8:	80e7a023          	sw	a4,-2048(a5) # 1a108800 <__l1_end+0xa1017e8>
1c00a8ac:	1c000437          	lui	s0,0x1c000
1c00a8b0:	262d                	jal	1c00abda <__rt_utils_init>
1c00a8b2:	57840413          	addi	s0,s0,1400 # 1c000578 <ctor_list+0x4>
1c00a8b6:	a12ff0ef          	jal	ra,1c009ac8 <__rt_allocs_init>
1c00a8ba:	718000ef          	jal	ra,1c00afd2 <__rt_thread_sched_init>
1c00a8be:	f2dfe0ef          	jal	ra,1c0097ea <__rt_event_sched_init>
1c00a8c2:	143000ef          	jal	ra,1c00b204 <__rt_padframe_init>
1c00a8c6:	0044278b          	p.lw	a5,4(s0!)
1c00a8ca:	e7d9                	bnez	a5,1c00a958 <__rt_init+0x15c>
1c00a8cc:	30045073          	csrwi	mstatus,8
1c00a8d0:	4501                	li	a0,0
1c00a8d2:	2ce1                	jal	1c00abaa <__rt_cbsys_exec>
1c00a8d4:	e531                	bnez	a0,1c00a920 <__rt_init+0x124>
1c00a8d6:	f14027f3          	csrr	a5,mhartid
1c00a8da:	8795                	srai	a5,a5,0x5
1c00a8dc:	f267b7b3          	p.bclr	a5,a5,25,6
1c00a8e0:	477d                	li	a4,31
1c00a8e2:	0ae78d63          	beq	a5,a4,1c00a99c <__rt_init+0x1a0>
1c00a8e6:	4681                	li	a3,0
1c00a8e8:	4601                	li	a2,0
1c00a8ea:	4581                	li	a1,0
1c00a8ec:	4505                	li	a0,1
1c00a8ee:	c7bd                	beqz	a5,1c00a95c <__rt_init+0x160>
1c00a8f0:	3e79                	jal	1c00a48e <rt_cluster_mount>
1c00a8f2:	6595                	lui	a1,0x5
1c00a8f4:	80058593          	addi	a1,a1,-2048 # 4800 <__rt_hyper_pending_tasks_last+0x4298>
1c00a8f8:	450d                	li	a0,3
1c00a8fa:	90eff0ef          	jal	ra,1c009a08 <rt_alloc>
1c00a8fe:	872a                	mv	a4,a0
1c00a900:	c105                	beqz	a0,1c00a920 <__rt_init+0x124>
1c00a902:	6805                	lui	a6,0x1
1c00a904:	80080813          	addi	a6,a6,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c00a908:	1c00a637          	lui	a2,0x1c00a
1c00a90c:	c002                	sw	zero,0(sp)
1c00a90e:	48a5                	li	a7,9
1c00a910:	87c2                	mv	a5,a6
1c00a912:	4681                	li	a3,0
1c00a914:	7d860613          	addi	a2,a2,2008 # 1c00a7d8 <cluster_start>
1c00a918:	4581                	li	a1,0
1c00a91a:	4501                	li	a0,0
1c00a91c:	34cd                	jal	1c00a3fe <rt_cluster_call>
1c00a91e:	cd3d                	beqz	a0,1c00a99c <__rt_init+0x1a0>
1c00a920:	f1402673          	csrr	a2,mhartid
1c00a924:	1c001537          	lui	a0,0x1c001
1c00a928:	40565593          	srai	a1,a2,0x5
1c00a92c:	f265b5b3          	p.bclr	a1,a1,25,6
1c00a930:	f4563633          	p.bclr	a2,a2,26,5
1c00a934:	ad050513          	addi	a0,a0,-1328 # 1c000ad0 <PIo2+0x174>
1c00a938:	02e010ef          	jal	ra,1c00b966 <printf>
1c00a93c:	7b9000ef          	jal	ra,1c00b8f4 <abort>
1c00a940:	002047b7          	lui	a5,0x204
1c00a944:	04000737          	lui	a4,0x4000
1c00a948:	cbd8                	sw	a4,20(a5)
1c00a94a:	b789                	j	1c00a88c <__rt_init+0x90>
1c00a94c:	002017b7          	lui	a5,0x201
1c00a950:	577d                	li	a4,-1
1c00a952:	40e7a023          	sw	a4,1024(a5) # 201400 <__l1_heap_size+0x1e8418>
1c00a956:	bf99                	j	1c00a8ac <__rt_init+0xb0>
1c00a958:	9782                	jalr	a5
1c00a95a:	b7b5                	j	1c00a8c6 <__rt_init+0xca>
1c00a95c:	3e0d                	jal	1c00a48e <rt_cluster_mount>
1c00a95e:	6591                	lui	a1,0x4
1c00a960:	450d                	li	a0,3
1c00a962:	8a6ff0ef          	jal	ra,1c009a08 <rt_alloc>
1c00a966:	dd4d                	beqz	a0,1c00a920 <__rt_init+0x124>
1c00a968:	00204737          	lui	a4,0x204
1c00a96c:	1ff00793          	li	a5,511
1c00a970:	08f72223          	sw	a5,132(a4) # 204084 <__l1_heap_size+0x1eb09c>
1c00a974:	1c0107b7          	lui	a5,0x1c010
1c00a978:	15678793          	addi	a5,a5,342 # 1c010156 <__rt_set_slave_stack>
1c00a97c:	c007c7b3          	p.bset	a5,a5,0,0
1c00a980:	08f72023          	sw	a5,128(a4)
1c00a984:	6785                	lui	a5,0x1
1c00a986:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c00a98a:	08f72023          	sw	a5,128(a4)
1c00a98e:	08a72023          	sw	a0,128(a4)
1c00a992:	4462                	lw	s0,24(sp)
1c00a994:	40f2                	lw	ra,28(sp)
1c00a996:	4501                	li	a0,0
1c00a998:	6105                	addi	sp,sp,32
1c00a99a:	bd3d                	j	1c00a7d8 <cluster_start>
1c00a99c:	40f2                	lw	ra,28(sp)
1c00a99e:	4462                	lw	s0,24(sp)
1c00a9a0:	6105                	addi	sp,sp,32
1c00a9a2:	8082                	ret

1c00a9a4 <__rt_deinit>:
1c00a9a4:	1c0017b7          	lui	a5,0x1c001
1c00a9a8:	6447a783          	lw	a5,1604(a5) # 1c001644 <__rt_platform>
1c00a9ac:	1141                	addi	sp,sp,-16
1c00a9ae:	c606                	sw	ra,12(sp)
1c00a9b0:	c422                	sw	s0,8(sp)
1c00a9b2:	0037b463          	p.bneimm	a5,3,1c00a9ba <__rt_deinit+0x16>
1c00a9b6:	7d005073          	csrwi	0x7d0,0
1c00a9ba:	4505                	li	a0,1
1c00a9bc:	1c000437          	lui	s0,0x1c000
1c00a9c0:	22ed                	jal	1c00abaa <__rt_cbsys_exec>
1c00a9c2:	5b040413          	addi	s0,s0,1456 # 1c0005b0 <dtor_list+0x4>
1c00a9c6:	0044278b          	p.lw	a5,4(s0!)
1c00a9ca:	e789                	bnez	a5,1c00a9d4 <__rt_deinit+0x30>
1c00a9cc:	40b2                	lw	ra,12(sp)
1c00a9ce:	4422                	lw	s0,8(sp)
1c00a9d0:	0141                	addi	sp,sp,16
1c00a9d2:	8082                	ret
1c00a9d4:	9782                	jalr	a5
1c00a9d6:	bfc5                	j	1c00a9c6 <__rt_deinit+0x22>

1c00a9d8 <pi_open_from_conf>:
1c00a9d8:	c14c                	sw	a1,4(a0)
1c00a9da:	8082                	ret

1c00a9dc <rt_irq_set_handler>:
1c00a9dc:	f14027f3          	csrr	a5,mhartid
1c00a9e0:	477d                	li	a4,31
1c00a9e2:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a9e6:	02e79e63          	bne	a5,a4,1c00aa22 <rt_irq_set_handler+0x46>
1c00a9ea:	30502773          	csrr	a4,mtvec
1c00a9ee:	c0073733          	p.bclr	a4,a4,0,0
1c00a9f2:	050a                	slli	a0,a0,0x2
1c00a9f4:	8d89                	sub	a1,a1,a0
1c00a9f6:	8d99                	sub	a1,a1,a4
1c00a9f8:	c14586b3          	p.extract	a3,a1,0,20
1c00a9fc:	06f00793          	li	a5,111
1c00aa00:	c1f6a7b3          	p.insert	a5,a3,0,31
1c00aa04:	d21586b3          	p.extract	a3,a1,9,1
1c00aa08:	d356a7b3          	p.insert	a5,a3,9,21
1c00aa0c:	c0b586b3          	p.extract	a3,a1,0,11
1c00aa10:	c146a7b3          	p.insert	a5,a3,0,20
1c00aa14:	cec585b3          	p.extract	a1,a1,7,12
1c00aa18:	cec5a7b3          	p.insert	a5,a1,7,12
1c00aa1c:	00f56723          	p.sw	a5,a4(a0)
1c00aa20:	8082                	ret
1c00aa22:	002007b7          	lui	a5,0x200
1c00aa26:	43b8                	lw	a4,64(a5)
1c00aa28:	b7e9                	j	1c00a9f2 <rt_irq_set_handler+0x16>

1c00aa2a <illegal_insn_handler_c>:
1c00aa2a:	8082                	ret

1c00aa2c <__rt_handle_illegal_instr>:
1c00aa2c:	1c0017b7          	lui	a5,0x1c001
1c00aa30:	47c7a703          	lw	a4,1148(a5) # 1c00147c <__rt_debug_config>
1c00aa34:	1141                	addi	sp,sp,-16
1c00aa36:	c422                	sw	s0,8(sp)
1c00aa38:	c606                	sw	ra,12(sp)
1c00aa3a:	fc173733          	p.bclr	a4,a4,30,1
1c00aa3e:	843e                	mv	s0,a5
1c00aa40:	c315                	beqz	a4,1c00aa64 <__rt_handle_illegal_instr+0x38>
1c00aa42:	341026f3          	csrr	a3,mepc
1c00aa46:	f1402673          	csrr	a2,mhartid
1c00aa4a:	1c001537          	lui	a0,0x1c001
1c00aa4e:	4298                	lw	a4,0(a3)
1c00aa50:	40565593          	srai	a1,a2,0x5
1c00aa54:	f265b5b3          	p.bclr	a1,a1,25,6
1c00aa58:	f4563633          	p.bclr	a2,a2,26,5
1c00aa5c:	b2850513          	addi	a0,a0,-1240 # 1c000b28 <PIo2+0x1cc>
1c00aa60:	707000ef          	jal	ra,1c00b966 <printf>
1c00aa64:	47c42783          	lw	a5,1148(s0)
1c00aa68:	c01797b3          	p.extractu	a5,a5,0,1
1c00aa6c:	c399                	beqz	a5,1c00aa72 <__rt_handle_illegal_instr+0x46>
1c00aa6e:	687000ef          	jal	ra,1c00b8f4 <abort>
1c00aa72:	4422                	lw	s0,8(sp)
1c00aa74:	40b2                	lw	ra,12(sp)
1c00aa76:	0141                	addi	sp,sp,16
1c00aa78:	fb3ff06f          	j	1c00aa2a <illegal_insn_handler_c>

1c00aa7c <__rt_irq_init>:
1c00aa7c:	f14027f3          	csrr	a5,mhartid
1c00aa80:	477d                	li	a4,31
1c00aa82:	ca5797b3          	p.extractu	a5,a5,5,5
1c00aa86:	02e79463          	bne	a5,a4,1c00aaae <__rt_irq_init+0x32>
1c00aa8a:	1a1097b7          	lui	a5,0x1a109
1c00aa8e:	577d                	li	a4,-1
1c00aa90:	c798                	sw	a4,8(a5)
1c00aa92:	f14027f3          	csrr	a5,mhartid
1c00aa96:	477d                	li	a4,31
1c00aa98:	ca5797b3          	p.extractu	a5,a5,5,5
1c00aa9c:	00e79e63          	bne	a5,a4,1c00aab8 <__rt_irq_init+0x3c>
1c00aaa0:	1c0087b7          	lui	a5,0x1c008
1c00aaa4:	00078793          	mv	a5,a5
1c00aaa8:	30579073          	csrw	mtvec,a5
1c00aaac:	8082                	ret
1c00aaae:	002047b7          	lui	a5,0x204
1c00aab2:	577d                	li	a4,-1
1c00aab4:	cb98                	sw	a4,16(a5)
1c00aab6:	bff1                	j	1c00aa92 <__rt_irq_init+0x16>
1c00aab8:	1c0087b7          	lui	a5,0x1c008
1c00aabc:	00200737          	lui	a4,0x200
1c00aac0:	00078793          	mv	a5,a5
1c00aac4:	c33c                	sw	a5,64(a4)
1c00aac6:	8082                	ret

1c00aac8 <__rt_fc_cluster_lock_req>:
1c00aac8:	1141                	addi	sp,sp,-16
1c00aaca:	c606                	sw	ra,12(sp)
1c00aacc:	c422                	sw	s0,8(sp)
1c00aace:	c226                	sw	s1,4(sp)
1c00aad0:	300474f3          	csrrci	s1,mstatus,8
1c00aad4:	09654703          	lbu	a4,150(a0)
1c00aad8:	411c                	lw	a5,0(a0)
1c00aada:	c721                	beqz	a4,1c00ab22 <__rt_fc_cluster_lock_req+0x5a>
1c00aadc:	4398                	lw	a4,0(a5)
1c00aade:	c30d                	beqz	a4,1c00ab00 <__rt_fc_cluster_lock_req+0x38>
1c00aae0:	43d8                	lw	a4,4(a5)
1c00aae2:	cf09                	beqz	a4,1c00aafc <__rt_fc_cluster_lock_req+0x34>
1c00aae4:	4798                	lw	a4,8(a5)
1c00aae6:	c348                	sw	a0,4(a4)
1c00aae8:	c788                	sw	a0,8(a5)
1c00aaea:	00052223          	sw	zero,4(a0)
1c00aaee:	30049073          	csrw	mstatus,s1
1c00aaf2:	40b2                	lw	ra,12(sp)
1c00aaf4:	4422                	lw	s0,8(sp)
1c00aaf6:	4492                	lw	s1,4(sp)
1c00aaf8:	0141                	addi	sp,sp,16
1c00aafa:	8082                	ret
1c00aafc:	c3c8                	sw	a0,4(a5)
1c00aafe:	b7ed                	j	1c00aae8 <__rt_fc_cluster_lock_req+0x20>
1c00ab00:	4705                	li	a4,1
1c00ab02:	08e50a23          	sb	a4,148(a0)
1c00ab06:	4705                	li	a4,1
1c00ab08:	c398                	sw	a4,0(a5)
1c00ab0a:	09554783          	lbu	a5,149(a0)
1c00ab0e:	04078793          	addi	a5,a5,64 # 1c008040 <__irq_vector_base+0x40>
1c00ab12:	00201737          	lui	a4,0x201
1c00ab16:	07da                	slli	a5,a5,0x16
1c00ab18:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e7e1c>
1c00ab1c:	0007e723          	p.sw	zero,a4(a5)
1c00ab20:	b7f9                	j	1c00aaee <__rt_fc_cluster_lock_req+0x26>
1c00ab22:	842a                	mv	s0,a0
1c00ab24:	47c8                	lw	a0,12(a5)
1c00ab26:	cd01                	beqz	a0,1c00ab3e <__rt_fc_cluster_lock_req+0x76>
1c00ab28:	0007a023          	sw	zero,0(a5)
1c00ab2c:	0007a623          	sw	zero,12(a5)
1c00ab30:	2171                	jal	1c00afbc <__rt_thread_wakeup>
1c00ab32:	4785                	li	a5,1
1c00ab34:	08f40a23          	sb	a5,148(s0)
1c00ab38:	09544783          	lbu	a5,149(s0)
1c00ab3c:	bfc9                	j	1c00ab0e <__rt_fc_cluster_lock_req+0x46>
1c00ab3e:	43d8                	lw	a4,4(a5)
1c00ab40:	e701                	bnez	a4,1c00ab48 <__rt_fc_cluster_lock_req+0x80>
1c00ab42:	0007a023          	sw	zero,0(a5)
1c00ab46:	b7f5                	j	1c00ab32 <__rt_fc_cluster_lock_req+0x6a>
1c00ab48:	4354                	lw	a3,4(a4)
1c00ab4a:	c3d4                	sw	a3,4(a5)
1c00ab4c:	4785                	li	a5,1
1c00ab4e:	08f70a23          	sb	a5,148(a4)
1c00ab52:	09574783          	lbu	a5,149(a4)
1c00ab56:	00201737          	lui	a4,0x201
1c00ab5a:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e7e1c>
1c00ab5e:	04078793          	addi	a5,a5,64
1c00ab62:	07da                	slli	a5,a5,0x16
1c00ab64:	0007e723          	p.sw	zero,a4(a5)
1c00ab68:	b7e9                	j	1c00ab32 <__rt_fc_cluster_lock_req+0x6a>

1c00ab6a <__rt_cbsys_add>:
1c00ab6a:	1101                	addi	sp,sp,-32
1c00ab6c:	cc22                	sw	s0,24(sp)
1c00ab6e:	ca26                	sw	s1,20(sp)
1c00ab70:	842a                	mv	s0,a0
1c00ab72:	84ae                	mv	s1,a1
1c00ab74:	4501                	li	a0,0
1c00ab76:	45b1                	li	a1,12
1c00ab78:	c632                	sw	a2,12(sp)
1c00ab7a:	ce06                	sw	ra,28(sp)
1c00ab7c:	e8dfe0ef          	jal	ra,1c009a08 <rt_alloc>
1c00ab80:	4632                	lw	a2,12(sp)
1c00ab82:	c115                	beqz	a0,1c00aba6 <__rt_cbsys_add+0x3c>
1c00ab84:	1c0017b7          	lui	a5,0x1c001
1c00ab88:	040a                	slli	s0,s0,0x2
1c00ab8a:	48078793          	addi	a5,a5,1152 # 1c001480 <cbsys_first>
1c00ab8e:	97a2                	add	a5,a5,s0
1c00ab90:	4398                	lw	a4,0(a5)
1c00ab92:	c104                	sw	s1,0(a0)
1c00ab94:	c150                	sw	a2,4(a0)
1c00ab96:	c518                	sw	a4,8(a0)
1c00ab98:	c388                	sw	a0,0(a5)
1c00ab9a:	4501                	li	a0,0
1c00ab9c:	40f2                	lw	ra,28(sp)
1c00ab9e:	4462                	lw	s0,24(sp)
1c00aba0:	44d2                	lw	s1,20(sp)
1c00aba2:	6105                	addi	sp,sp,32
1c00aba4:	8082                	ret
1c00aba6:	557d                	li	a0,-1
1c00aba8:	bfd5                	j	1c00ab9c <__rt_cbsys_add+0x32>

1c00abaa <__rt_cbsys_exec>:
1c00abaa:	1141                	addi	sp,sp,-16
1c00abac:	c422                	sw	s0,8(sp)
1c00abae:	1c001437          	lui	s0,0x1c001
1c00abb2:	050a                	slli	a0,a0,0x2
1c00abb4:	48040413          	addi	s0,s0,1152 # 1c001480 <cbsys_first>
1c00abb8:	20a47403          	p.lw	s0,a0(s0)
1c00abbc:	c606                	sw	ra,12(sp)
1c00abbe:	e411                	bnez	s0,1c00abca <__rt_cbsys_exec+0x20>
1c00abc0:	4501                	li	a0,0
1c00abc2:	40b2                	lw	ra,12(sp)
1c00abc4:	4422                	lw	s0,8(sp)
1c00abc6:	0141                	addi	sp,sp,16
1c00abc8:	8082                	ret
1c00abca:	401c                	lw	a5,0(s0)
1c00abcc:	4048                	lw	a0,4(s0)
1c00abce:	9782                	jalr	a5
1c00abd0:	e119                	bnez	a0,1c00abd6 <__rt_cbsys_exec+0x2c>
1c00abd2:	4400                	lw	s0,8(s0)
1c00abd4:	b7ed                	j	1c00abbe <__rt_cbsys_exec+0x14>
1c00abd6:	557d                	li	a0,-1
1c00abd8:	b7ed                	j	1c00abc2 <__rt_cbsys_exec+0x18>

1c00abda <__rt_utils_init>:
1c00abda:	1c0017b7          	lui	a5,0x1c001
1c00abde:	48078793          	addi	a5,a5,1152 # 1c001480 <cbsys_first>
1c00abe2:	0007a023          	sw	zero,0(a5)
1c00abe6:	0007a223          	sw	zero,4(a5)
1c00abea:	0007a423          	sw	zero,8(a5)
1c00abee:	0007a623          	sw	zero,12(a5)
1c00abf2:	0007a823          	sw	zero,16(a5)
1c00abf6:	0007aa23          	sw	zero,20(a5)
1c00abfa:	8082                	ret

1c00abfc <__rt_fc_lock>:
1c00abfc:	1141                	addi	sp,sp,-16
1c00abfe:	c422                	sw	s0,8(sp)
1c00ac00:	842a                	mv	s0,a0
1c00ac02:	c606                	sw	ra,12(sp)
1c00ac04:	c226                	sw	s1,4(sp)
1c00ac06:	c04a                	sw	s2,0(sp)
1c00ac08:	300474f3          	csrrci	s1,mstatus,8
1c00ac0c:	401c                	lw	a5,0(s0)
1c00ac0e:	eb99                	bnez	a5,1c00ac24 <__rt_fc_lock+0x28>
1c00ac10:	4785                	li	a5,1
1c00ac12:	c01c                	sw	a5,0(s0)
1c00ac14:	30049073          	csrw	mstatus,s1
1c00ac18:	40b2                	lw	ra,12(sp)
1c00ac1a:	4422                	lw	s0,8(sp)
1c00ac1c:	4492                	lw	s1,4(sp)
1c00ac1e:	4902                	lw	s2,0(sp)
1c00ac20:	0141                	addi	sp,sp,16
1c00ac22:	8082                	ret
1c00ac24:	04802783          	lw	a5,72(zero) # 48 <__rt_thread_current>
1c00ac28:	4585                	li	a1,1
1c00ac2a:	e3ff5517          	auipc	a0,0xe3ff5
1c00ac2e:	3e250513          	addi	a0,a0,994 # c <__rt_sched>
1c00ac32:	c45c                	sw	a5,12(s0)
1c00ac34:	b15fe0ef          	jal	ra,1c009748 <__rt_event_execute>
1c00ac38:	bfd1                	j	1c00ac0c <__rt_fc_lock+0x10>

1c00ac3a <__rt_fc_unlock>:
1c00ac3a:	300476f3          	csrrci	a3,mstatus,8
1c00ac3e:	415c                	lw	a5,4(a0)
1c00ac40:	e791                	bnez	a5,1c00ac4c <__rt_fc_unlock+0x12>
1c00ac42:	00052023          	sw	zero,0(a0)
1c00ac46:	30069073          	csrw	mstatus,a3
1c00ac4a:	8082                	ret
1c00ac4c:	43d8                	lw	a4,4(a5)
1c00ac4e:	c158                	sw	a4,4(a0)
1c00ac50:	4705                	li	a4,1
1c00ac52:	08e78a23          	sb	a4,148(a5)
1c00ac56:	0957c783          	lbu	a5,149(a5)
1c00ac5a:	00201737          	lui	a4,0x201
1c00ac5e:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e7e1c>
1c00ac62:	04078793          	addi	a5,a5,64
1c00ac66:	07da                	slli	a5,a5,0x16
1c00ac68:	0007e723          	p.sw	zero,a4(a5)
1c00ac6c:	bfe9                	j	1c00ac46 <__rt_fc_unlock+0xc>

1c00ac6e <__rt_fc_cluster_lock>:
1c00ac6e:	f14027f3          	csrr	a5,mhartid
1c00ac72:	8795                	srai	a5,a5,0x5
1c00ac74:	f267b7b3          	p.bclr	a5,a5,25,6
1c00ac78:	08f58aa3          	sb	a5,149(a1) # 4095 <__rt_hyper_pending_tasks_last+0x3b2d>
1c00ac7c:	4785                	li	a5,1
1c00ac7e:	08f58b23          	sb	a5,150(a1)
1c00ac82:	1c00b7b7          	lui	a5,0x1c00b
1c00ac86:	ac878793          	addi	a5,a5,-1336 # 1c00aac8 <__rt_fc_cluster_lock_req>
1c00ac8a:	c188                	sw	a0,0(a1)
1c00ac8c:	08058a23          	sb	zero,148(a1)
1c00ac90:	0205a423          	sw	zero,40(a1)
1c00ac94:	0205a623          	sw	zero,44(a1)
1c00ac98:	c59c                	sw	a5,8(a1)
1c00ac9a:	c5cc                	sw	a1,12(a1)
1c00ac9c:	05a1                	addi	a1,a1,8
1c00ac9e:	c005c533          	p.bset	a0,a1,0,0
1c00aca2:	e96ff06f          	j	1c00a338 <__rt_cluster_push_fc_event>

1c00aca6 <__rt_fc_cluster_unlock>:
1c00aca6:	f14027f3          	csrr	a5,mhartid
1c00acaa:	8795                	srai	a5,a5,0x5
1c00acac:	f267b7b3          	p.bclr	a5,a5,25,6
1c00acb0:	08f58aa3          	sb	a5,149(a1)
1c00acb4:	1c00b7b7          	lui	a5,0x1c00b
1c00acb8:	ac878793          	addi	a5,a5,-1336 # 1c00aac8 <__rt_fc_cluster_lock_req>
1c00acbc:	c188                	sw	a0,0(a1)
1c00acbe:	08058a23          	sb	zero,148(a1)
1c00acc2:	08058b23          	sb	zero,150(a1)
1c00acc6:	0205a423          	sw	zero,40(a1)
1c00acca:	0205a623          	sw	zero,44(a1)
1c00acce:	c59c                	sw	a5,8(a1)
1c00acd0:	c5cc                	sw	a1,12(a1)
1c00acd2:	05a1                	addi	a1,a1,8
1c00acd4:	c005c533          	p.bset	a0,a1,0,0
1c00acd8:	e60ff06f          	j	1c00a338 <__rt_cluster_push_fc_event>

1c00acdc <__rt_event_enqueue>:
1c00acdc:	00c02783          	lw	a5,12(zero) # c <__rt_sched>
1c00ace0:	00052c23          	sw	zero,24(a0)
1c00ace4:	c799                	beqz	a5,1c00acf2 <__rt_event_enqueue+0x16>
1c00ace6:	01002783          	lw	a5,16(zero) # 10 <__rt_sched+0x4>
1c00acea:	cf88                	sw	a0,24(a5)
1c00acec:	00a02823          	sw	a0,16(zero) # 10 <__rt_sched+0x4>
1c00acf0:	8082                	ret
1c00acf2:	00a02623          	sw	a0,12(zero) # c <__rt_sched>
1c00acf6:	bfdd                	j	1c00acec <__rt_event_enqueue+0x10>

1c00acf8 <__rt_bridge_check_bridge_req.part.5>:
1c00acf8:	1c001737          	lui	a4,0x1c001
1c00acfc:	58470793          	addi	a5,a4,1412 # 1c001584 <__hal_debug_struct>
1c00ad00:	0a47a783          	lw	a5,164(a5)
1c00ad04:	58470713          	addi	a4,a4,1412
1c00ad08:	c789                	beqz	a5,1c00ad12 <__rt_bridge_check_bridge_req.part.5+0x1a>
1c00ad0a:	4f94                	lw	a3,24(a5)
1c00ad0c:	e681                	bnez	a3,1c00ad14 <__rt_bridge_check_bridge_req.part.5+0x1c>
1c00ad0e:	0af72623          	sw	a5,172(a4)
1c00ad12:	8082                	ret
1c00ad14:	479c                	lw	a5,8(a5)
1c00ad16:	bfcd                	j	1c00ad08 <__rt_bridge_check_bridge_req.part.5+0x10>

1c00ad18 <__rt_bridge_wait>:
1c00ad18:	f14027f3          	csrr	a5,mhartid
1c00ad1c:	477d                	li	a4,31
1c00ad1e:	ca5797b3          	p.extractu	a5,a5,5,5
1c00ad22:	02e79e63          	bne	a5,a4,1c00ad5e <__rt_bridge_wait+0x46>
1c00ad26:	1a1097b7          	lui	a5,0x1a109
1c00ad2a:	00c78513          	addi	a0,a5,12 # 1a10900c <__l1_end+0xa101ff4>
1c00ad2e:	6711                	lui	a4,0x4
1c00ad30:	00478593          	addi	a1,a5,4
1c00ad34:	00878613          	addi	a2,a5,8
1c00ad38:	300476f3          	csrrci	a3,mstatus,8
1c00ad3c:	00052803          	lw	a6,0(a0)
1c00ad40:	01181893          	slli	a7,a6,0x11
1c00ad44:	0008c963          	bltz	a7,1c00ad56 <__rt_bridge_wait+0x3e>
1c00ad48:	c198                	sw	a4,0(a1)
1c00ad4a:	10500073          	wfi
1c00ad4e:	c218                	sw	a4,0(a2)
1c00ad50:	30069073          	csrw	mstatus,a3
1c00ad54:	b7d5                	j	1c00ad38 <__rt_bridge_wait+0x20>
1c00ad56:	07d1                	addi	a5,a5,20
1c00ad58:	c398                	sw	a4,0(a5)
1c00ad5a:	30069073          	csrw	mstatus,a3
1c00ad5e:	8082                	ret

1c00ad60 <__rt_bridge_handle_notif>:
1c00ad60:	1141                	addi	sp,sp,-16
1c00ad62:	c422                	sw	s0,8(sp)
1c00ad64:	1c001437          	lui	s0,0x1c001
1c00ad68:	58440793          	addi	a5,s0,1412 # 1c001584 <__hal_debug_struct>
1c00ad6c:	0a47a783          	lw	a5,164(a5)
1c00ad70:	c606                	sw	ra,12(sp)
1c00ad72:	c226                	sw	s1,4(sp)
1c00ad74:	c04a                	sw	s2,0(sp)
1c00ad76:	58440413          	addi	s0,s0,1412
1c00ad7a:	c399                	beqz	a5,1c00ad80 <__rt_bridge_handle_notif+0x20>
1c00ad7c:	4bd8                	lw	a4,20(a5)
1c00ad7e:	e30d                	bnez	a4,1c00ada0 <__rt_bridge_handle_notif+0x40>
1c00ad80:	0b442783          	lw	a5,180(s0)
1c00ad84:	c789                	beqz	a5,1c00ad8e <__rt_bridge_handle_notif+0x2e>
1c00ad86:	43a8                	lw	a0,64(a5)
1c00ad88:	0a042a23          	sw	zero,180(s0)
1c00ad8c:	3f81                	jal	1c00acdc <__rt_event_enqueue>
1c00ad8e:	0ac42783          	lw	a5,172(s0)
1c00ad92:	eb95                	bnez	a5,1c00adc6 <__rt_bridge_handle_notif+0x66>
1c00ad94:	4422                	lw	s0,8(sp)
1c00ad96:	40b2                	lw	ra,12(sp)
1c00ad98:	4492                	lw	s1,4(sp)
1c00ad9a:	4902                	lw	s2,0(sp)
1c00ad9c:	0141                	addi	sp,sp,16
1c00ad9e:	bfa9                	j	1c00acf8 <__rt_bridge_check_bridge_req.part.5>
1c00ada0:	4784                	lw	s1,8(a5)
1c00ada2:	4fd8                	lw	a4,28(a5)
1c00ada4:	0a942223          	sw	s1,164(s0)
1c00ada8:	cb01                	beqz	a4,1c00adb8 <__rt_bridge_handle_notif+0x58>
1c00adaa:	0b042703          	lw	a4,176(s0)
1c00adae:	c798                	sw	a4,8(a5)
1c00adb0:	0af42823          	sw	a5,176(s0)
1c00adb4:	87a6                	mv	a5,s1
1c00adb6:	b7d1                	j	1c00ad7a <__rt_bridge_handle_notif+0x1a>
1c00adb8:	43a8                	lw	a0,64(a5)
1c00adba:	30047973          	csrrci	s2,mstatus,8
1c00adbe:	3f39                	jal	1c00acdc <__rt_event_enqueue>
1c00adc0:	30091073          	csrw	mstatus,s2
1c00adc4:	bfc5                	j	1c00adb4 <__rt_bridge_handle_notif+0x54>
1c00adc6:	40b2                	lw	ra,12(sp)
1c00adc8:	4422                	lw	s0,8(sp)
1c00adca:	4492                	lw	s1,4(sp)
1c00adcc:	4902                	lw	s2,0(sp)
1c00adce:	0141                	addi	sp,sp,16
1c00add0:	8082                	ret

1c00add2 <__rt_bridge_check_connection>:
1c00add2:	1c001737          	lui	a4,0x1c001
1c00add6:	58470713          	addi	a4,a4,1412 # 1c001584 <__hal_debug_struct>
1c00adda:	471c                	lw	a5,8(a4)
1c00addc:	ef8d                	bnez	a5,1c00ae16 <__rt_bridge_check_connection+0x44>
1c00adde:	1a1047b7          	lui	a5,0x1a104
1c00ade2:	07478793          	addi	a5,a5,116 # 1a104074 <__l1_end+0xa0fd05c>
1c00ade6:	4394                	lw	a3,0(a5)
1c00ade8:	cc9696b3          	p.extractu	a3,a3,6,9
1c00adec:	0276b563          	p.bneimm	a3,7,1c00ae16 <__rt_bridge_check_connection+0x44>
1c00adf0:	1141                	addi	sp,sp,-16
1c00adf2:	c422                	sw	s0,8(sp)
1c00adf4:	c606                	sw	ra,12(sp)
1c00adf6:	4685                	li	a3,1
1c00adf8:	c714                	sw	a3,8(a4)
1c00adfa:	4709                	li	a4,2
1c00adfc:	c398                	sw	a4,0(a5)
1c00adfe:	843e                	mv	s0,a5
1c00ae00:	401c                	lw	a5,0(s0)
1c00ae02:	cc9797b3          	p.extractu	a5,a5,6,9
1c00ae06:	0077a663          	p.beqimm	a5,7,1c00ae12 <__rt_bridge_check_connection+0x40>
1c00ae0a:	40b2                	lw	ra,12(sp)
1c00ae0c:	4422                	lw	s0,8(sp)
1c00ae0e:	0141                	addi	sp,sp,16
1c00ae10:	8082                	ret
1c00ae12:	3719                	jal	1c00ad18 <__rt_bridge_wait>
1c00ae14:	b7f5                	j	1c00ae00 <__rt_bridge_check_connection+0x2e>
1c00ae16:	8082                	ret

1c00ae18 <__rt_bridge_set_available>:
1c00ae18:	1c0017b7          	lui	a5,0x1c001
1c00ae1c:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00ae20:	4798                	lw	a4,8(a5)
1c00ae22:	1a1047b7          	lui	a5,0x1a104
1c00ae26:	07478793          	addi	a5,a5,116 # 1a104074 <__l1_end+0xa0fd05c>
1c00ae2a:	e701                	bnez	a4,1c00ae32 <__rt_bridge_set_available+0x1a>
1c00ae2c:	4721                	li	a4,8
1c00ae2e:	c398                	sw	a4,0(a5)
1c00ae30:	8082                	ret
1c00ae32:	4709                	li	a4,2
1c00ae34:	bfed                	j	1c00ae2e <__rt_bridge_set_available+0x16>

1c00ae36 <__rt_bridge_send_notif>:
1c00ae36:	1141                	addi	sp,sp,-16
1c00ae38:	c606                	sw	ra,12(sp)
1c00ae3a:	3f61                	jal	1c00add2 <__rt_bridge_check_connection>
1c00ae3c:	1c0017b7          	lui	a5,0x1c001
1c00ae40:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00ae44:	479c                	lw	a5,8(a5)
1c00ae46:	c789                	beqz	a5,1c00ae50 <__rt_bridge_send_notif+0x1a>
1c00ae48:	1a1047b7          	lui	a5,0x1a104
1c00ae4c:	4719                	li	a4,6
1c00ae4e:	dbf8                	sw	a4,116(a5)
1c00ae50:	40b2                	lw	ra,12(sp)
1c00ae52:	0141                	addi	sp,sp,16
1c00ae54:	8082                	ret

1c00ae56 <__rt_bridge_clear_notif>:
1c00ae56:	1141                	addi	sp,sp,-16
1c00ae58:	c606                	sw	ra,12(sp)
1c00ae5a:	3fa5                	jal	1c00add2 <__rt_bridge_check_connection>
1c00ae5c:	1c0017b7          	lui	a5,0x1c001
1c00ae60:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00ae64:	479c                	lw	a5,8(a5)
1c00ae66:	c781                	beqz	a5,1c00ae6e <__rt_bridge_clear_notif+0x18>
1c00ae68:	40b2                	lw	ra,12(sp)
1c00ae6a:	0141                	addi	sp,sp,16
1c00ae6c:	b775                	j	1c00ae18 <__rt_bridge_set_available>
1c00ae6e:	40b2                	lw	ra,12(sp)
1c00ae70:	0141                	addi	sp,sp,16
1c00ae72:	8082                	ret

1c00ae74 <__rt_bridge_printf_flush>:
1c00ae74:	1141                	addi	sp,sp,-16
1c00ae76:	c422                	sw	s0,8(sp)
1c00ae78:	c606                	sw	ra,12(sp)
1c00ae7a:	1c001437          	lui	s0,0x1c001
1c00ae7e:	3f91                	jal	1c00add2 <__rt_bridge_check_connection>
1c00ae80:	58440793          	addi	a5,s0,1412 # 1c001584 <__hal_debug_struct>
1c00ae84:	479c                	lw	a5,8(a5)
1c00ae86:	c385                	beqz	a5,1c00aea6 <__rt_bridge_printf_flush+0x32>
1c00ae88:	58440413          	addi	s0,s0,1412
1c00ae8c:	485c                	lw	a5,20(s0)
1c00ae8e:	e399                	bnez	a5,1c00ae94 <__rt_bridge_printf_flush+0x20>
1c00ae90:	4c1c                	lw	a5,24(s0)
1c00ae92:	cb91                	beqz	a5,1c00aea6 <__rt_bridge_printf_flush+0x32>
1c00ae94:	374d                	jal	1c00ae36 <__rt_bridge_send_notif>
1c00ae96:	485c                	lw	a5,20(s0)
1c00ae98:	e789                	bnez	a5,1c00aea2 <__rt_bridge_printf_flush+0x2e>
1c00ae9a:	4422                	lw	s0,8(sp)
1c00ae9c:	40b2                	lw	ra,12(sp)
1c00ae9e:	0141                	addi	sp,sp,16
1c00aea0:	bf5d                	j	1c00ae56 <__rt_bridge_clear_notif>
1c00aea2:	3d9d                	jal	1c00ad18 <__rt_bridge_wait>
1c00aea4:	bfcd                	j	1c00ae96 <__rt_bridge_printf_flush+0x22>
1c00aea6:	40b2                	lw	ra,12(sp)
1c00aea8:	4422                	lw	s0,8(sp)
1c00aeaa:	0141                	addi	sp,sp,16
1c00aeac:	8082                	ret

1c00aeae <__rt_bridge_req_shutdown>:
1c00aeae:	1141                	addi	sp,sp,-16
1c00aeb0:	c606                	sw	ra,12(sp)
1c00aeb2:	c422                	sw	s0,8(sp)
1c00aeb4:	3f39                	jal	1c00add2 <__rt_bridge_check_connection>
1c00aeb6:	1c0017b7          	lui	a5,0x1c001
1c00aeba:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00aebe:	479c                	lw	a5,8(a5)
1c00aec0:	c3a9                	beqz	a5,1c00af02 <__rt_bridge_req_shutdown+0x54>
1c00aec2:	1a104437          	lui	s0,0x1a104
1c00aec6:	377d                	jal	1c00ae74 <__rt_bridge_printf_flush>
1c00aec8:	07440413          	addi	s0,s0,116 # 1a104074 <__l1_end+0xa0fd05c>
1c00aecc:	401c                	lw	a5,0(s0)
1c00aece:	cc9797b3          	p.extractu	a5,a5,6,9
1c00aed2:	0277ac63          	p.beqimm	a5,7,1c00af0a <__rt_bridge_req_shutdown+0x5c>
1c00aed6:	4791                	li	a5,4
1c00aed8:	c01c                	sw	a5,0(s0)
1c00aeda:	1a104437          	lui	s0,0x1a104
1c00aede:	07440413          	addi	s0,s0,116 # 1a104074 <__l1_end+0xa0fd05c>
1c00aee2:	401c                	lw	a5,0(s0)
1c00aee4:	cc9797b3          	p.extractu	a5,a5,6,9
1c00aee8:	0277b363          	p.bneimm	a5,7,1c00af0e <__rt_bridge_req_shutdown+0x60>
1c00aeec:	00042023          	sw	zero,0(s0)
1c00aef0:	1a104437          	lui	s0,0x1a104
1c00aef4:	07440413          	addi	s0,s0,116 # 1a104074 <__l1_end+0xa0fd05c>
1c00aef8:	401c                	lw	a5,0(s0)
1c00aefa:	cc9797b3          	p.extractu	a5,a5,6,9
1c00aefe:	0077aa63          	p.beqimm	a5,7,1c00af12 <__rt_bridge_req_shutdown+0x64>
1c00af02:	40b2                	lw	ra,12(sp)
1c00af04:	4422                	lw	s0,8(sp)
1c00af06:	0141                	addi	sp,sp,16
1c00af08:	8082                	ret
1c00af0a:	3539                	jal	1c00ad18 <__rt_bridge_wait>
1c00af0c:	b7c1                	j	1c00aecc <__rt_bridge_req_shutdown+0x1e>
1c00af0e:	3529                	jal	1c00ad18 <__rt_bridge_wait>
1c00af10:	bfc9                	j	1c00aee2 <__rt_bridge_req_shutdown+0x34>
1c00af12:	3519                	jal	1c00ad18 <__rt_bridge_wait>
1c00af14:	b7d5                	j	1c00aef8 <__rt_bridge_req_shutdown+0x4a>

1c00af16 <__rt_bridge_init>:
1c00af16:	1c0017b7          	lui	a5,0x1c001
1c00af1a:	1a109737          	lui	a4,0x1a109
1c00af1e:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00af22:	0741                	addi	a4,a4,16
1c00af24:	0ae7ac23          	sw	a4,184(a5)
1c00af28:	4741                	li	a4,16
1c00af2a:	0a07a223          	sw	zero,164(a5)
1c00af2e:	0a07a623          	sw	zero,172(a5)
1c00af32:	0a07aa23          	sw	zero,180(a5)
1c00af36:	0ae7ae23          	sw	a4,188(a5)
1c00af3a:	02c00793          	li	a5,44
1c00af3e:	0007a823          	sw	zero,16(a5)
1c00af42:	0007a023          	sw	zero,0(a5)
1c00af46:	8082                	ret

1c00af48 <__rt_thread_enqueue_ready>:
1c00af48:	04002703          	lw	a4,64(zero) # 40 <__rt_ready_queue>
1c00af4c:	02052c23          	sw	zero,56(a0)
1c00af50:	04000793          	li	a5,64
1c00af54:	e711                	bnez	a4,1c00af60 <__rt_thread_enqueue_ready+0x18>
1c00af56:	c388                	sw	a0,0(a5)
1c00af58:	c3c8                	sw	a0,4(a5)
1c00af5a:	0c052a23          	sw	zero,212(a0)
1c00af5e:	8082                	ret
1c00af60:	43d8                	lw	a4,4(a5)
1c00af62:	df08                	sw	a0,56(a4)
1c00af64:	bfd5                	j	1c00af58 <__rt_thread_enqueue_ready+0x10>

1c00af66 <__rt_thread_sleep>:
1c00af66:	04000713          	li	a4,64
1c00af6a:	4708                	lw	a0,8(a4)
1c00af6c:	04000793          	li	a5,64
1c00af70:	438c                	lw	a1,0(a5)
1c00af72:	c999                	beqz	a1,1c00af88 <__rt_thread_sleep+0x22>
1c00af74:	5d98                	lw	a4,56(a1)
1c00af76:	c398                	sw	a4,0(a5)
1c00af78:	4705                	li	a4,1
1c00af7a:	0ce5aa23          	sw	a4,212(a1)
1c00af7e:	00b50c63          	beq	a0,a1,1c00af96 <__rt_thread_sleep+0x30>
1c00af82:	c78c                	sw	a1,8(a5)
1c00af84:	e69fd06f          	j	1c008dec <__rt_thread_switch>
1c00af88:	10500073          	wfi
1c00af8c:	30045073          	csrwi	mstatus,8
1c00af90:	30047773          	csrrci	a4,mstatus,8
1c00af94:	bff1                	j	1c00af70 <__rt_thread_sleep+0xa>
1c00af96:	8082                	ret

1c00af98 <rt_thread_exit>:
1c00af98:	300477f3          	csrrci	a5,mstatus,8
1c00af9c:	04802783          	lw	a5,72(zero) # 48 <__rt_thread_current>
1c00afa0:	4705                	li	a4,1
1c00afa2:	c3e8                	sw	a0,68(a5)
1c00afa4:	5fc8                	lw	a0,60(a5)
1c00afa6:	c3b8                	sw	a4,64(a5)
1c00afa8:	c909                	beqz	a0,1c00afba <rt_thread_exit+0x22>
1c00afaa:	0d452783          	lw	a5,212(a0)
1c00afae:	c791                	beqz	a5,1c00afba <rt_thread_exit+0x22>
1c00afb0:	1141                	addi	sp,sp,-16
1c00afb2:	c606                	sw	ra,12(sp)
1c00afb4:	3f51                	jal	1c00af48 <__rt_thread_enqueue_ready>
1c00afb6:	40b2                	lw	ra,12(sp)
1c00afb8:	0141                	addi	sp,sp,16
1c00afba:	b775                	j	1c00af66 <__rt_thread_sleep>

1c00afbc <__rt_thread_wakeup>:
1c00afbc:	5d18                	lw	a4,56(a0)
1c00afbe:	eb09                	bnez	a4,1c00afd0 <__rt_thread_wakeup+0x14>
1c00afc0:	04802703          	lw	a4,72(zero) # 48 <__rt_thread_current>
1c00afc4:	00a70663          	beq	a4,a0,1c00afd0 <__rt_thread_wakeup+0x14>
1c00afc8:	0d452783          	lw	a5,212(a0)
1c00afcc:	c391                	beqz	a5,1c00afd0 <__rt_thread_wakeup+0x14>
1c00afce:	bfad                	j	1c00af48 <__rt_thread_enqueue_ready>
1c00afd0:	8082                	ret

1c00afd2 <__rt_thread_sched_init>:
1c00afd2:	1141                	addi	sp,sp,-16
1c00afd4:	c422                	sw	s0,8(sp)
1c00afd6:	1c0097b7          	lui	a5,0x1c009
1c00afda:	1c001437          	lui	s0,0x1c001
1c00afde:	c226                	sw	s1,4(sp)
1c00afe0:	c04a                	sw	s2,0(sp)
1c00afe2:	c606                	sw	ra,12(sp)
1c00afe4:	49840413          	addi	s0,s0,1176 # 1c001498 <__rt_thread_main>
1c00afe8:	de678793          	addi	a5,a5,-538 # 1c008de6 <__rt_thread_start>
1c00afec:	c01c                	sw	a5,0(s0)
1c00afee:	1c00b7b7          	lui	a5,0x1c00b
1c00aff2:	04840913          	addi	s2,s0,72
1c00aff6:	f9878793          	addi	a5,a5,-104 # 1c00af98 <rt_thread_exit>
1c00affa:	04000493          	li	s1,64
1c00affe:	c45c                	sw	a5,12(s0)
1c00b000:	854a                	mv	a0,s2
1c00b002:	4785                	li	a5,1
1c00b004:	e3ff5597          	auipc	a1,0xe3ff5
1c00b008:	00858593          	addi	a1,a1,8 # c <__rt_sched>
1c00b00c:	0cf42a23          	sw	a5,212(s0)
1c00b010:	0004a023          	sw	zero,0(s1)
1c00b014:	02042a23          	sw	zero,52(s0)
1c00b018:	00042223          	sw	zero,4(s0)
1c00b01c:	00042423          	sw	zero,8(s0)
1c00b020:	e28fe0ef          	jal	ra,1c009648 <__rt_event_init>
1c00b024:	00802783          	lw	a5,8(zero) # 8 <__rt_first_free>
1c00b028:	c480                	sw	s0,8(s1)
1c00b02a:	40b2                	lw	ra,12(sp)
1c00b02c:	d03c                	sw	a5,96(s0)
1c00b02e:	4422                	lw	s0,8(sp)
1c00b030:	01202423          	sw	s2,8(zero) # 8 <__rt_first_free>
1c00b034:	4492                	lw	s1,4(sp)
1c00b036:	4902                	lw	s2,0(sp)
1c00b038:	0141                	addi	sp,sp,16
1c00b03a:	8082                	ret

1c00b03c <__rt_fll_set_freq>:
1c00b03c:	1101                	addi	sp,sp,-32
1c00b03e:	cc22                	sw	s0,24(sp)
1c00b040:	ce06                	sw	ra,28(sp)
1c00b042:	842a                	mv	s0,a0
1c00b044:	00153563          	p.bneimm	a0,1,1c00b04e <__rt_fll_set_freq+0x12>
1c00b048:	c62e                	sw	a1,12(sp)
1c00b04a:	3595                	jal	1c00aeae <__rt_bridge_req_shutdown>
1c00b04c:	45b2                	lw	a1,12(sp)
1c00b04e:	10059733          	p.fl1	a4,a1
1c00b052:	47f5                	li	a5,29
1c00b054:	4505                	li	a0,1
1c00b056:	82e7b7db          	p.subun	a5,a5,a4,1
1c00b05a:	04f567b3          	p.max	a5,a0,a5
1c00b05e:	fff78713          	addi	a4,a5,-1
1c00b062:	00f595b3          	sll	a1,a1,a5
1c00b066:	00e51533          	sll	a0,a0,a4
1c00b06a:	1c0016b7          	lui	a3,0x1c001
1c00b06e:	dc05b733          	p.bclr	a4,a1,14,0
1c00b072:	c0f7255b          	p.addunr	a0,a4,a5
1c00b076:	7dc68693          	addi	a3,a3,2012 # 1c0017dc <__rt_fll_freq>
1c00b07a:	00241713          	slli	a4,s0,0x2
1c00b07e:	00a6e723          	p.sw	a0,a4(a3)
1c00b082:	1c001737          	lui	a4,0x1c001
1c00b086:	74470713          	addi	a4,a4,1860 # 1c001744 <__rt_fll_is_on>
1c00b08a:	9722                	add	a4,a4,s0
1c00b08c:	00074703          	lbu	a4,0(a4)
1c00b090:	cf19                	beqz	a4,1c00b0ae <__rt_fll_set_freq+0x72>
1c00b092:	0412                	slli	s0,s0,0x4
1c00b094:	0411                	addi	s0,s0,4
1c00b096:	1a1006b7          	lui	a3,0x1a100
1c00b09a:	2086f703          	p.lw	a4,s0(a3)
1c00b09e:	81bd                	srli	a1,a1,0xf
1c00b0a0:	de05a733          	p.insert	a4,a1,15,0
1c00b0a4:	0785                	addi	a5,a5,1
1c00b0a6:	c7a7a733          	p.insert	a4,a5,3,26
1c00b0aa:	00e6e423          	p.sw	a4,s0(a3)
1c00b0ae:	40f2                	lw	ra,28(sp)
1c00b0b0:	4462                	lw	s0,24(sp)
1c00b0b2:	6105                	addi	sp,sp,32
1c00b0b4:	8082                	ret

1c00b0b6 <__rt_fll_init>:
1c00b0b6:	1141                	addi	sp,sp,-16
1c00b0b8:	00451613          	slli	a2,a0,0x4
1c00b0bc:	c226                	sw	s1,4(sp)
1c00b0be:	00460813          	addi	a6,a2,4
1c00b0c2:	84aa                	mv	s1,a0
1c00b0c4:	1a1006b7          	lui	a3,0x1a100
1c00b0c8:	c606                	sw	ra,12(sp)
1c00b0ca:	c422                	sw	s0,8(sp)
1c00b0cc:	2106f703          	p.lw	a4,a6(a3)
1c00b0d0:	87ba                	mv	a5,a4
1c00b0d2:	04074163          	bltz	a4,1c00b114 <__rt_fll_init+0x5e>
1c00b0d6:	00860893          	addi	a7,a2,8
1c00b0da:	2116f503          	p.lw	a0,a7(a3)
1c00b0de:	4599                	li	a1,6
1c00b0e0:	caa5a533          	p.insert	a0,a1,5,10
1c00b0e4:	05000593          	li	a1,80
1c00b0e8:	d705a533          	p.insert	a0,a1,11,16
1c00b0ec:	1a1005b7          	lui	a1,0x1a100
1c00b0f0:	00a5e8a3          	p.sw	a0,a7(a1)
1c00b0f4:	0631                	addi	a2,a2,12
1c00b0f6:	20c6f683          	p.lw	a3,a2(a3)
1c00b0fa:	14c00513          	li	a0,332
1c00b0fe:	d30526b3          	p.insert	a3,a0,9,16
1c00b102:	00d5e623          	p.sw	a3,a2(a1)
1c00b106:	4685                	li	a3,1
1c00b108:	c1e6a7b3          	p.insert	a5,a3,0,30
1c00b10c:	c1f6a7b3          	p.insert	a5,a3,0,31
1c00b110:	00f5e823          	p.sw	a5,a6(a1)
1c00b114:	1c001637          	lui	a2,0x1c001
1c00b118:	00249693          	slli	a3,s1,0x2
1c00b11c:	7dc60613          	addi	a2,a2,2012 # 1c0017dc <__rt_fll_freq>
1c00b120:	96b2                	add	a3,a3,a2
1c00b122:	4280                	lw	s0,0(a3)
1c00b124:	c00d                	beqz	s0,1c00b146 <__rt_fll_init+0x90>
1c00b126:	85a2                	mv	a1,s0
1c00b128:	8526                	mv	a0,s1
1c00b12a:	3f09                	jal	1c00b03c <__rt_fll_set_freq>
1c00b12c:	1c0017b7          	lui	a5,0x1c001
1c00b130:	74478793          	addi	a5,a5,1860 # 1c001744 <__rt_fll_is_on>
1c00b134:	4705                	li	a4,1
1c00b136:	00e7c4a3          	p.sb	a4,s1(a5)
1c00b13a:	8522                	mv	a0,s0
1c00b13c:	40b2                	lw	ra,12(sp)
1c00b13e:	4422                	lw	s0,8(sp)
1c00b140:	4492                	lw	s1,4(sp)
1c00b142:	0141                	addi	sp,sp,16
1c00b144:	8082                	ret
1c00b146:	10075733          	p.exthz	a4,a4
1c00b14a:	c7a797b3          	p.extractu	a5,a5,3,26
1c00b14e:	073e                	slli	a4,a4,0xf
1c00b150:	17fd                	addi	a5,a5,-1
1c00b152:	00f75433          	srl	s0,a4,a5
1c00b156:	c280                	sw	s0,0(a3)
1c00b158:	bfd1                	j	1c00b12c <__rt_fll_init+0x76>

1c00b15a <__rt_fll_deinit>:
1c00b15a:	1c0017b7          	lui	a5,0x1c001
1c00b15e:	74478793          	addi	a5,a5,1860 # 1c001744 <__rt_fll_is_on>
1c00b162:	0007c523          	p.sb	zero,a0(a5)
1c00b166:	8082                	ret

1c00b168 <__rt_flls_constructor>:
1c00b168:	1c0017b7          	lui	a5,0x1c001
1c00b16c:	7c07ae23          	sw	zero,2012(a5) # 1c0017dc <__rt_fll_freq>
1c00b170:	7dc78793          	addi	a5,a5,2012
1c00b174:	0007a223          	sw	zero,4(a5)
1c00b178:	0007a423          	sw	zero,8(a5)
1c00b17c:	1c0017b7          	lui	a5,0x1c001
1c00b180:	74478793          	addi	a5,a5,1860 # 1c001744 <__rt_fll_is_on>
1c00b184:	00079023          	sh	zero,0(a5)
1c00b188:	00078123          	sb	zero,2(a5)
1c00b18c:	8082                	ret

1c00b18e <rt_freq_set_and_get>:
1c00b18e:	1101                	addi	sp,sp,-32
1c00b190:	cc22                	sw	s0,24(sp)
1c00b192:	c84a                	sw	s2,16(sp)
1c00b194:	842a                	mv	s0,a0
1c00b196:	892e                	mv	s2,a1
1c00b198:	ce06                	sw	ra,28(sp)
1c00b19a:	ca26                	sw	s1,20(sp)
1c00b19c:	300474f3          	csrrci	s1,mstatus,8
1c00b1a0:	c632                	sw	a2,12(sp)
1c00b1a2:	3d69                	jal	1c00b03c <__rt_fll_set_freq>
1c00b1a4:	4632                	lw	a2,12(sp)
1c00b1a6:	c211                	beqz	a2,1c00b1aa <rt_freq_set_and_get+0x1c>
1c00b1a8:	c208                	sw	a0,0(a2)
1c00b1aa:	1c0017b7          	lui	a5,0x1c001
1c00b1ae:	040a                	slli	s0,s0,0x2
1c00b1b0:	7e878793          	addi	a5,a5,2024 # 1c0017e8 <__rt_freq_domains>
1c00b1b4:	0127e423          	p.sw	s2,s0(a5)
1c00b1b8:	30049073          	csrw	mstatus,s1
1c00b1bc:	40f2                	lw	ra,28(sp)
1c00b1be:	4462                	lw	s0,24(sp)
1c00b1c0:	44d2                	lw	s1,20(sp)
1c00b1c2:	4942                	lw	s2,16(sp)
1c00b1c4:	4501                	li	a0,0
1c00b1c6:	6105                	addi	sp,sp,32
1c00b1c8:	8082                	ret

1c00b1ca <__rt_freq_init>:
1c00b1ca:	1141                	addi	sp,sp,-16
1c00b1cc:	c606                	sw	ra,12(sp)
1c00b1ce:	c422                	sw	s0,8(sp)
1c00b1d0:	c226                	sw	s1,4(sp)
1c00b1d2:	3f59                	jal	1c00b168 <__rt_flls_constructor>
1c00b1d4:	1c0014b7          	lui	s1,0x1c001
1c00b1d8:	4505                	li	a0,1
1c00b1da:	3df1                	jal	1c00b0b6 <__rt_fll_init>
1c00b1dc:	7e848413          	addi	s0,s1,2024 # 1c0017e8 <__rt_freq_domains>
1c00b1e0:	c048                	sw	a0,4(s0)
1c00b1e2:	4501                	li	a0,0
1c00b1e4:	3dc9                	jal	1c00b0b6 <__rt_fll_init>
1c00b1e6:	7ea4a423          	sw	a0,2024(s1)
1c00b1ea:	4509                	li	a0,2
1c00b1ec:	35e9                	jal	1c00b0b6 <__rt_fll_init>
1c00b1ee:	4795                	li	a5,5
1c00b1f0:	1a104737          	lui	a4,0x1a104
1c00b1f4:	c408                	sw	a0,8(s0)
1c00b1f6:	0cf72823          	sw	a5,208(a4) # 1a1040d0 <__l1_end+0xa0fd0b8>
1c00b1fa:	40b2                	lw	ra,12(sp)
1c00b1fc:	4422                	lw	s0,8(sp)
1c00b1fe:	4492                	lw	s1,4(sp)
1c00b200:	0141                	addi	sp,sp,16
1c00b202:	8082                	ret

1c00b204 <__rt_padframe_init>:
1c00b204:	300477f3          	csrrci	a5,mstatus,8
1c00b208:	30079073          	csrw	mstatus,a5
1c00b20c:	8082                	ret

1c00b20e <rt_periph_copy>:
1c00b20e:	7179                	addi	sp,sp,-48
1c00b210:	d422                	sw	s0,40(sp)
1c00b212:	842a                	mv	s0,a0
1c00b214:	d606                	sw	ra,44(sp)
1c00b216:	d226                	sw	s1,36(sp)
1c00b218:	d04a                	sw	s2,32(sp)
1c00b21a:	30047973          	csrrci	s2,mstatus,8
1c00b21e:	4015d493          	srai	s1,a1,0x1
1c00b222:	1a102537          	lui	a0,0x1a102
1c00b226:	08050513          	addi	a0,a0,128 # 1a102080 <__l1_end+0xa0fb068>
1c00b22a:	049e                	slli	s1,s1,0x7
1c00b22c:	94aa                	add	s1,s1,a0
1c00b22e:	00459513          	slli	a0,a1,0x4
1c00b232:	8941                	andi	a0,a0,16
1c00b234:	94aa                	add	s1,s1,a0
1c00b236:	853e                	mv	a0,a5
1c00b238:	ef89                	bnez	a5,1c00b252 <rt_periph_copy+0x44>
1c00b23a:	ce2e                	sw	a1,28(sp)
1c00b23c:	cc32                	sw	a2,24(sp)
1c00b23e:	ca36                	sw	a3,20(sp)
1c00b240:	c83a                	sw	a4,16(sp)
1c00b242:	c63e                	sw	a5,12(sp)
1c00b244:	c16fe0ef          	jal	ra,1c00965a <__rt_wait_event_prepare_blocking>
1c00b248:	47b2                	lw	a5,12(sp)
1c00b24a:	4742                	lw	a4,16(sp)
1c00b24c:	46d2                	lw	a3,20(sp)
1c00b24e:	4662                	lw	a2,24(sp)
1c00b250:	45f2                	lw	a1,28(sp)
1c00b252:	e419                	bnez	s0,1c00b260 <rt_periph_copy+0x52>
1c00b254:	03450413          	addi	s0,a0,52
1c00b258:	04052023          	sw	zero,64(a0)
1c00b25c:	04052823          	sw	zero,80(a0)
1c00b260:	00c42803          	lw	a6,12(s0)
1c00b264:	c054                	sw	a3,4(s0)
1c00b266:	cc08                	sw	a0,24(s0)
1c00b268:	f6483833          	p.bclr	a6,a6,27,4
1c00b26c:	4891                	li	a7,4
1c00b26e:	c0474733          	p.bset	a4,a4,0,4
1c00b272:	0908e063          	bltu	a7,a6,1c00b2f2 <rt_periph_copy+0xe4>
1c00b276:	04c00893          	li	a7,76
1c00b27a:	0596                	slli	a1,a1,0x5
1c00b27c:	98ae                	add	a7,a7,a1
1c00b27e:	0008a303          	lw	t1,0(a7)
1c00b282:	00042a23          	sw	zero,20(s0)
1c00b286:	04c00813          	li	a6,76
1c00b28a:	04031463          	bnez	t1,1c00b2d2 <rt_periph_copy+0xc4>
1c00b28e:	0088a023          	sw	s0,0(a7)
1c00b292:	00b808b3          	add	a7,a6,a1
1c00b296:	0088a303          	lw	t1,8(a7)
1c00b29a:	0088a223          	sw	s0,4(a7)
1c00b29e:	02031f63          	bnez	t1,1c00b2dc <rt_periph_copy+0xce>
1c00b2a2:	00848e13          	addi	t3,s1,8
1c00b2a6:	000e2883          	lw	a7,0(t3)
1c00b2aa:	0208f893          	andi	a7,a7,32
1c00b2ae:	02089763          	bnez	a7,1c00b2dc <rt_periph_copy+0xce>
1c00b2b2:	00c4a22b          	p.sw	a2,4(s1!)
1c00b2b6:	c094                	sw	a3,0(s1)
1c00b2b8:	00ee2023          	sw	a4,0(t3)
1c00b2bc:	e399                	bnez	a5,1c00b2c2 <rt_periph_copy+0xb4>
1c00b2be:	ceafe0ef          	jal	ra,1c0097a8 <__rt_wait_event>
1c00b2c2:	30091073          	csrw	mstatus,s2
1c00b2c6:	50b2                	lw	ra,44(sp)
1c00b2c8:	5422                	lw	s0,40(sp)
1c00b2ca:	5492                	lw	s1,36(sp)
1c00b2cc:	5902                	lw	s2,32(sp)
1c00b2ce:	6145                	addi	sp,sp,48
1c00b2d0:	8082                	ret
1c00b2d2:	0048a883          	lw	a7,4(a7)
1c00b2d6:	0088aa23          	sw	s0,20(a7)
1c00b2da:	bf65                	j	1c00b292 <rt_periph_copy+0x84>
1c00b2dc:	00042823          	sw	zero,16(s0)
1c00b2e0:	c010                	sw	a2,0(s0)
1c00b2e2:	c054                	sw	a3,4(s0)
1c00b2e4:	c418                	sw	a4,8(s0)
1c00b2e6:	fc031be3          	bnez	t1,1c00b2bc <rt_periph_copy+0xae>
1c00b2ea:	982e                	add	a6,a6,a1
1c00b2ec:	00882423          	sw	s0,8(a6)
1c00b2f0:	b7f1                	j	1c00b2bc <rt_periph_copy+0xae>
1c00b2f2:	fc6835e3          	p.bneimm	a6,6,1c00b2bc <rt_periph_copy+0xae>
1c00b2f6:	04c00893          	li	a7,76
1c00b2fa:	0596                	slli	a1,a1,0x5
1c00b2fc:	98ae                	add	a7,a7,a1
1c00b2fe:	0008a303          	lw	t1,0(a7)
1c00b302:	00042a23          	sw	zero,20(s0)
1c00b306:	04c00813          	li	a6,76
1c00b30a:	02031563          	bnez	t1,1c00b334 <rt_periph_copy+0x126>
1c00b30e:	0088a023          	sw	s0,0(a7)
1c00b312:	95c2                	add	a1,a1,a6
1c00b314:	c1c0                	sw	s0,4(a1)
1c00b316:	02031463          	bnez	t1,1c00b33e <rt_periph_copy+0x130>
1c00b31a:	02442803          	lw	a6,36(s0)
1c00b31e:	1a1025b7          	lui	a1,0x1a102
1c00b322:	4b05a023          	sw	a6,1184(a1) # 1a1024a0 <__l1_end+0xa0fb488>
1c00b326:	85a6                	mv	a1,s1
1c00b328:	00c5a22b          	p.sw	a2,4(a1!)
1c00b32c:	c194                	sw	a3,0(a1)
1c00b32e:	04a1                	addi	s1,s1,8
1c00b330:	c098                	sw	a4,0(s1)
1c00b332:	b769                	j	1c00b2bc <rt_periph_copy+0xae>
1c00b334:	0048a883          	lw	a7,4(a7)
1c00b338:	0088aa23          	sw	s0,20(a7)
1c00b33c:	bfd9                	j	1c00b312 <rt_periph_copy+0x104>
1c00b33e:	c418                	sw	a4,8(s0)
1c00b340:	4598                	lw	a4,8(a1)
1c00b342:	c010                	sw	a2,0(s0)
1c00b344:	c054                	sw	a3,4(s0)
1c00b346:	00042823          	sw	zero,16(s0)
1c00b34a:	fb2d                	bnez	a4,1c00b2bc <rt_periph_copy+0xae>
1c00b34c:	c580                	sw	s0,8(a1)
1c00b34e:	b7bd                	j	1c00b2bc <rt_periph_copy+0xae>

1c00b350 <__rt_periph_init>:
1c00b350:	04c00693          	li	a3,76
1c00b354:	1c009637          	lui	a2,0x1c009
1c00b358:	42068693          	addi	a3,a3,1056 # 1a100420 <__l1_end+0xa0f9408>
1c00b35c:	04c00713          	li	a4,76
1c00b360:	ec260613          	addi	a2,a2,-318 # 1c008ec2 <udma_event_handler>
1c00b364:	010250fb          	lp.setupi	x1,16,1c00b36c <__rt_periph_init+0x1c>
1c00b368:	00c6a22b          	p.sw	a2,4(a3!)
1c00b36c:	0001                	nop
1c00b36e:	40072023          	sw	zero,1024(a4)
1c00b372:	40072223          	sw	zero,1028(a4)
1c00b376:	40072423          	sw	zero,1032(a4)
1c00b37a:	40072623          	sw	zero,1036(a4)
1c00b37e:	40072823          	sw	zero,1040(a4)
1c00b382:	40072a23          	sw	zero,1044(a4)
1c00b386:	40072c23          	sw	zero,1048(a4)
1c00b38a:	1a102837          	lui	a6,0x1a102
1c00b38e:	40072e23          	sw	zero,1052(a4)
1c00b392:	04c00793          	li	a5,76
1c00b396:	4681                	li	a3,0
1c00b398:	08080813          	addi	a6,a6,128 # 1a102080 <__l1_end+0xa0fb068>
1c00b39c:	0208d0fb          	lp.setupi	x1,32,1c00b3be <__rt_periph_init+0x6e>
1c00b3a0:	4016d713          	srai	a4,a3,0x1
1c00b3a4:	00469513          	slli	a0,a3,0x4
1c00b3a8:	071e                	slli	a4,a4,0x7
1c00b3aa:	9742                	add	a4,a4,a6
1c00b3ac:	8941                	andi	a0,a0,16
1c00b3ae:	972a                	add	a4,a4,a0
1c00b3b0:	0007a023          	sw	zero,0(a5)
1c00b3b4:	0007a423          	sw	zero,8(a5)
1c00b3b8:	c7d8                	sw	a4,12(a5)
1c00b3ba:	cfd0                	sw	a2,28(a5)
1c00b3bc:	0685                	addi	a3,a3,1
1c00b3be:	02078793          	addi	a5,a5,32
1c00b3c2:	8082                	ret

1c00b3c4 <__rt_i2c_init>:
1c00b3c4:	1c0107b7          	lui	a5,0x1c010
1c00b3c8:	18078223          	sb	zero,388(a5) # 1c010184 <__cluster_text_end+0x4>
1c00b3cc:	8082                	ret

1c00b3ce <__rt_rtc_init>:
1c00b3ce:	4ec00793          	li	a5,1260
1c00b3d2:	0207ac23          	sw	zero,56(a5)
1c00b3d6:	02078823          	sb	zero,48(a5)
1c00b3da:	0207aa23          	sw	zero,52(a5)
1c00b3de:	8082                	ret

1c00b3e0 <__rt_hyper_init>:
1c00b3e0:	1c001737          	lui	a4,0x1c001
1c00b3e4:	52800793          	li	a5,1320
1c00b3e8:	74072423          	sw	zero,1864(a4) # 1c001748 <__pi_hyper_cluster_reqs_first>
1c00b3ec:	577d                	li	a4,-1
1c00b3ee:	0007aa23          	sw	zero,20(a5)
1c00b3f2:	0207a823          	sw	zero,48(a5)
1c00b3f6:	cf98                	sw	a4,24(a5)
1c00b3f8:	8082                	ret

1c00b3fa <rt_is_fc>:
1c00b3fa:	f1402573          	csrr	a0,mhartid
1c00b3fe:	8515                	srai	a0,a0,0x5
1c00b400:	f2653533          	p.bclr	a0,a0,25,6
1c00b404:	1505                	addi	a0,a0,-31
1c00b406:	00153513          	seqz	a0,a0
1c00b40a:	8082                	ret

1c00b40c <__rt_io_end_of_flush>:
1c00b40c:	1c0017b7          	lui	a5,0x1c001
1c00b410:	7407a823          	sw	zero,1872(a5) # 1c001750 <__rt_io_pending_flush>
1c00b414:	00052c23          	sw	zero,24(a0)
1c00b418:	8082                	ret

1c00b41a <__rt_io_uart_wait_req>:
1c00b41a:	1141                	addi	sp,sp,-16
1c00b41c:	c226                	sw	s1,4(sp)
1c00b41e:	84aa                	mv	s1,a0
1c00b420:	c606                	sw	ra,12(sp)
1c00b422:	c422                	sw	s0,8(sp)
1c00b424:	c04a                	sw	s2,0(sp)
1c00b426:	30047973          	csrrci	s2,mstatus,8
1c00b42a:	1c001437          	lui	s0,0x1c001
1c00b42e:	74c40413          	addi	s0,s0,1868 # 1c00174c <__rt_io_event_current>
1c00b432:	4008                	lw	a0,0(s0)
1c00b434:	c509                	beqz	a0,1c00b43e <__rt_io_uart_wait_req+0x24>
1c00b436:	b9cfe0ef          	jal	ra,1c0097d2 <rt_event_wait>
1c00b43a:	00042023          	sw	zero,0(s0)
1c00b43e:	4785                	li	a5,1
1c00b440:	08f48623          	sb	a5,140(s1)
1c00b444:	08d4c783          	lbu	a5,141(s1)
1c00b448:	00201737          	lui	a4,0x201
1c00b44c:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e7e1c>
1c00b450:	04078793          	addi	a5,a5,64
1c00b454:	07da                	slli	a5,a5,0x16
1c00b456:	0007e723          	p.sw	zero,a4(a5)
1c00b45a:	30091073          	csrw	mstatus,s2
1c00b45e:	40b2                	lw	ra,12(sp)
1c00b460:	4422                	lw	s0,8(sp)
1c00b462:	4492                	lw	s1,4(sp)
1c00b464:	4902                	lw	s2,0(sp)
1c00b466:	0141                	addi	sp,sp,16
1c00b468:	8082                	ret

1c00b46a <__rt_io_start>:
1c00b46a:	1101                	addi	sp,sp,-32
1c00b46c:	0028                	addi	a0,sp,8
1c00b46e:	ce06                	sw	ra,28(sp)
1c00b470:	cc22                	sw	s0,24(sp)
1c00b472:	006010ef          	jal	ra,1c00c478 <rt_uart_conf_init>
1c00b476:	4585                	li	a1,1
1c00b478:	4501                	li	a0,0
1c00b47a:	9f8fe0ef          	jal	ra,1c009672 <rt_event_alloc>
1c00b47e:	547d                	li	s0,-1
1c00b480:	ed1d                	bnez	a0,1c00b4be <__rt_io_start+0x54>
1c00b482:	1c0017b7          	lui	a5,0x1c001
1c00b486:	6487a783          	lw	a5,1608(a5) # 1c001648 <__rt_iodev_uart_baudrate>
1c00b48a:	842a                	mv	s0,a0
1c00b48c:	1c001537          	lui	a0,0x1c001
1c00b490:	e3ff5597          	auipc	a1,0xe3ff5
1c00b494:	b7c58593          	addi	a1,a1,-1156 # c <__rt_sched>
1c00b498:	67c50513          	addi	a0,a0,1660 # 1c00167c <__rt_io_event>
1c00b49c:	c43e                	sw	a5,8(sp)
1c00b49e:	9aafe0ef          	jal	ra,1c009648 <__rt_event_init>
1c00b4a2:	1c0017b7          	lui	a5,0x1c001
1c00b4a6:	75c7a503          	lw	a0,1884(a5) # 1c00175c <__rt_iodev_uart_channel>
1c00b4aa:	4681                	li	a3,0
1c00b4ac:	4601                	li	a2,0
1c00b4ae:	002c                	addi	a1,sp,8
1c00b4b0:	0511                	addi	a0,a0,4
1c00b4b2:	7d7000ef          	jal	ra,1c00c488 <__rt_uart_open>
1c00b4b6:	1c0017b7          	lui	a5,0x1c001
1c00b4ba:	74a7aa23          	sw	a0,1876(a5) # 1c001754 <_rt_io_uart>
1c00b4be:	8522                	mv	a0,s0
1c00b4c0:	40f2                	lw	ra,28(sp)
1c00b4c2:	4462                	lw	s0,24(sp)
1c00b4c4:	6105                	addi	sp,sp,32
1c00b4c6:	8082                	ret

1c00b4c8 <rt_event_execute.isra.2.constprop.11>:
1c00b4c8:	1141                	addi	sp,sp,-16
1c00b4ca:	c606                	sw	ra,12(sp)
1c00b4cc:	c422                	sw	s0,8(sp)
1c00b4ce:	30047473          	csrrci	s0,mstatus,8
1c00b4d2:	4585                	li	a1,1
1c00b4d4:	e3ff5517          	auipc	a0,0xe3ff5
1c00b4d8:	b3850513          	addi	a0,a0,-1224 # c <__rt_sched>
1c00b4dc:	a6cfe0ef          	jal	ra,1c009748 <__rt_event_execute>
1c00b4e0:	30041073          	csrw	mstatus,s0
1c00b4e4:	40b2                	lw	ra,12(sp)
1c00b4e6:	4422                	lw	s0,8(sp)
1c00b4e8:	0141                	addi	sp,sp,16
1c00b4ea:	8082                	ret

1c00b4ec <__rt_io_lock>:
1c00b4ec:	1c0017b7          	lui	a5,0x1c001
1c00b4f0:	5947a783          	lw	a5,1428(a5) # 1c001594 <__hal_debug_struct+0x10>
1c00b4f4:	c791                	beqz	a5,1c00b500 <__rt_io_lock+0x14>
1c00b4f6:	1c0017b7          	lui	a5,0x1c001
1c00b4fa:	7547a783          	lw	a5,1876(a5) # 1c001754 <_rt_io_uart>
1c00b4fe:	c3a1                	beqz	a5,1c00b53e <__rt_io_lock+0x52>
1c00b500:	7171                	addi	sp,sp,-176
1c00b502:	d706                	sw	ra,172(sp)
1c00b504:	3ddd                	jal	1c00b3fa <rt_is_fc>
1c00b506:	1c0017b7          	lui	a5,0x1c001
1c00b50a:	c901                	beqz	a0,1c00b51a <__rt_io_lock+0x2e>
1c00b50c:	57478513          	addi	a0,a5,1396 # 1c001574 <__rt_io_fc_lock>
1c00b510:	eecff0ef          	jal	ra,1c00abfc <__rt_fc_lock>
1c00b514:	50ba                	lw	ra,172(sp)
1c00b516:	614d                	addi	sp,sp,176
1c00b518:	8082                	ret
1c00b51a:	002c                	addi	a1,sp,8
1c00b51c:	57478513          	addi	a0,a5,1396
1c00b520:	f4eff0ef          	jal	ra,1c00ac6e <__rt_fc_cluster_lock>
1c00b524:	4689                	li	a3,2
1c00b526:	00204737          	lui	a4,0x204
1c00b52a:	09c14783          	lbu	a5,156(sp)
1c00b52e:	0ff7f793          	andi	a5,a5,255
1c00b532:	f3ed                	bnez	a5,1c00b514 <__rt_io_lock+0x28>
1c00b534:	c714                	sw	a3,8(a4)
1c00b536:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1eb054>
1c00b53a:	c354                	sw	a3,4(a4)
1c00b53c:	b7fd                	j	1c00b52a <__rt_io_lock+0x3e>
1c00b53e:	8082                	ret

1c00b540 <__rt_io_unlock>:
1c00b540:	1c0017b7          	lui	a5,0x1c001
1c00b544:	5947a783          	lw	a5,1428(a5) # 1c001594 <__hal_debug_struct+0x10>
1c00b548:	c791                	beqz	a5,1c00b554 <__rt_io_unlock+0x14>
1c00b54a:	1c0017b7          	lui	a5,0x1c001
1c00b54e:	7547a783          	lw	a5,1876(a5) # 1c001754 <_rt_io_uart>
1c00b552:	c3a1                	beqz	a5,1c00b592 <__rt_io_unlock+0x52>
1c00b554:	7171                	addi	sp,sp,-176
1c00b556:	d706                	sw	ra,172(sp)
1c00b558:	354d                	jal	1c00b3fa <rt_is_fc>
1c00b55a:	1c0017b7          	lui	a5,0x1c001
1c00b55e:	c901                	beqz	a0,1c00b56e <__rt_io_unlock+0x2e>
1c00b560:	57478513          	addi	a0,a5,1396 # 1c001574 <__rt_io_fc_lock>
1c00b564:	ed6ff0ef          	jal	ra,1c00ac3a <__rt_fc_unlock>
1c00b568:	50ba                	lw	ra,172(sp)
1c00b56a:	614d                	addi	sp,sp,176
1c00b56c:	8082                	ret
1c00b56e:	002c                	addi	a1,sp,8
1c00b570:	57478513          	addi	a0,a5,1396
1c00b574:	f32ff0ef          	jal	ra,1c00aca6 <__rt_fc_cluster_unlock>
1c00b578:	4689                	li	a3,2
1c00b57a:	00204737          	lui	a4,0x204
1c00b57e:	09c14783          	lbu	a5,156(sp)
1c00b582:	0ff7f793          	andi	a5,a5,255
1c00b586:	f3ed                	bnez	a5,1c00b568 <__rt_io_unlock+0x28>
1c00b588:	c714                	sw	a3,8(a4)
1c00b58a:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1eb054>
1c00b58e:	c354                	sw	a3,4(a4)
1c00b590:	b7fd                	j	1c00b57e <__rt_io_unlock+0x3e>
1c00b592:	8082                	ret

1c00b594 <__rt_io_uart_wait_pending>:
1c00b594:	7135                	addi	sp,sp,-160
1c00b596:	cd22                	sw	s0,152(sp)
1c00b598:	cf06                	sw	ra,156(sp)
1c00b59a:	cb26                	sw	s1,148(sp)
1c00b59c:	1c001437          	lui	s0,0x1c001
1c00b5a0:	75042783          	lw	a5,1872(s0) # 1c001750 <__rt_io_pending_flush>
1c00b5a4:	e39d                	bnez	a5,1c00b5ca <__rt_io_uart_wait_pending+0x36>
1c00b5a6:	1c001437          	lui	s0,0x1c001
1c00b5aa:	74c40413          	addi	s0,s0,1868 # 1c00174c <__rt_io_event_current>
1c00b5ae:	4004                	lw	s1,0(s0)
1c00b5b0:	c881                	beqz	s1,1c00b5c0 <__rt_io_uart_wait_pending+0x2c>
1c00b5b2:	35a1                	jal	1c00b3fa <rt_is_fc>
1c00b5b4:	cd19                	beqz	a0,1c00b5d2 <__rt_io_uart_wait_pending+0x3e>
1c00b5b6:	8526                	mv	a0,s1
1c00b5b8:	a1afe0ef          	jal	ra,1c0097d2 <rt_event_wait>
1c00b5bc:	00042023          	sw	zero,0(s0)
1c00b5c0:	40fa                	lw	ra,156(sp)
1c00b5c2:	446a                	lw	s0,152(sp)
1c00b5c4:	44da                	lw	s1,148(sp)
1c00b5c6:	610d                	addi	sp,sp,160
1c00b5c8:	8082                	ret
1c00b5ca:	3f9d                	jal	1c00b540 <__rt_io_unlock>
1c00b5cc:	3df5                	jal	1c00b4c8 <rt_event_execute.isra.2.constprop.11>
1c00b5ce:	3f39                	jal	1c00b4ec <__rt_io_lock>
1c00b5d0:	bfc1                	j	1c00b5a0 <__rt_io_uart_wait_pending+0xc>
1c00b5d2:	f14027f3          	csrr	a5,mhartid
1c00b5d6:	8795                	srai	a5,a5,0x5
1c00b5d8:	f267b7b3          	p.bclr	a5,a5,25,6
1c00b5dc:	08f106a3          	sb	a5,141(sp)
1c00b5e0:	1c00b7b7          	lui	a5,0x1c00b
1c00b5e4:	41a78793          	addi	a5,a5,1050 # 1c00b41a <__rt_io_uart_wait_req>
1c00b5e8:	c03e                	sw	a5,0(sp)
1c00b5ea:	00010793          	mv	a5,sp
1c00b5ee:	4705                	li	a4,1
1c00b5f0:	c23e                	sw	a5,4(sp)
1c00b5f2:	850a                	mv	a0,sp
1c00b5f4:	1c0017b7          	lui	a5,0x1c001
1c00b5f8:	68e7ae23          	sw	a4,1692(a5) # 1c00169c <__rt_io_event+0x20>
1c00b5fc:	08010623          	sb	zero,140(sp)
1c00b600:	d002                	sw	zero,32(sp)
1c00b602:	d202                	sw	zero,36(sp)
1c00b604:	d35fe0ef          	jal	ra,1c00a338 <__rt_cluster_push_fc_event>
1c00b608:	4689                	li	a3,2
1c00b60a:	00204737          	lui	a4,0x204
1c00b60e:	08c14783          	lbu	a5,140(sp)
1c00b612:	0ff7f793          	andi	a5,a5,255
1c00b616:	f7cd                	bnez	a5,1c00b5c0 <__rt_io_uart_wait_pending+0x2c>
1c00b618:	c714                	sw	a3,8(a4)
1c00b61a:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1eb054>
1c00b61e:	c354                	sw	a3,4(a4)
1c00b620:	b7fd                	j	1c00b60e <__rt_io_uart_wait_pending+0x7a>

1c00b622 <__rt_io_stop>:
1c00b622:	1141                	addi	sp,sp,-16
1c00b624:	c422                	sw	s0,8(sp)
1c00b626:	1c001437          	lui	s0,0x1c001
1c00b62a:	c606                	sw	ra,12(sp)
1c00b62c:	75440413          	addi	s0,s0,1876 # 1c001754 <_rt_io_uart>
1c00b630:	3795                	jal	1c00b594 <__rt_io_uart_wait_pending>
1c00b632:	4008                	lw	a0,0(s0)
1c00b634:	4581                	li	a1,0
1c00b636:	6d7000ef          	jal	ra,1c00c50c <rt_uart_close>
1c00b63a:	40b2                	lw	ra,12(sp)
1c00b63c:	00042023          	sw	zero,0(s0)
1c00b640:	4422                	lw	s0,8(sp)
1c00b642:	4501                	li	a0,0
1c00b644:	0141                	addi	sp,sp,16
1c00b646:	8082                	ret

1c00b648 <__rt_io_uart_flush.constprop.10>:
1c00b648:	7131                	addi	sp,sp,-192
1c00b64a:	dd22                	sw	s0,184(sp)
1c00b64c:	df06                	sw	ra,188(sp)
1c00b64e:	db26                	sw	s1,180(sp)
1c00b650:	d94a                	sw	s2,176(sp)
1c00b652:	d74e                	sw	s3,172(sp)
1c00b654:	d552                	sw	s4,168(sp)
1c00b656:	d356                	sw	s5,164(sp)
1c00b658:	1c001437          	lui	s0,0x1c001
1c00b65c:	75042783          	lw	a5,1872(s0) # 1c001750 <__rt_io_pending_flush>
1c00b660:	75040a13          	addi	s4,s0,1872
1c00b664:	e7bd                	bnez	a5,1c00b6d2 <__rt_io_uart_flush.constprop.10+0x8a>
1c00b666:	1c0014b7          	lui	s1,0x1c001
1c00b66a:	58448793          	addi	a5,s1,1412 # 1c001584 <__hal_debug_struct>
1c00b66e:	4f80                	lw	s0,24(a5)
1c00b670:	58448a93          	addi	s5,s1,1412
1c00b674:	c431                	beqz	s0,1c00b6c0 <__rt_io_uart_flush.constprop.10+0x78>
1c00b676:	3351                	jal	1c00b3fa <rt_is_fc>
1c00b678:	1c0017b7          	lui	a5,0x1c001
1c00b67c:	7547a903          	lw	s2,1876(a5) # 1c001754 <_rt_io_uart>
1c00b680:	1c0019b7          	lui	s3,0x1c001
1c00b684:	cd29                	beqz	a0,1c00b6de <__rt_io_uart_flush.constprop.10+0x96>
1c00b686:	1c00b5b7          	lui	a1,0x1c00b
1c00b68a:	4785                	li	a5,1
1c00b68c:	58448613          	addi	a2,s1,1412
1c00b690:	40c58593          	addi	a1,a1,1036 # 1c00b40c <__rt_io_end_of_flush>
1c00b694:	4501                	li	a0,0
1c00b696:	00fa2023          	sw	a5,0(s4)
1c00b69a:	84efe0ef          	jal	ra,1c0096e8 <rt_event_get>
1c00b69e:	00492583          	lw	a1,4(s2)
1c00b6a2:	87aa                	mv	a5,a0
1c00b6a4:	4701                	li	a4,0
1c00b6a6:	0586                	slli	a1,a1,0x1
1c00b6a8:	86a2                	mv	a3,s0
1c00b6aa:	5a098613          	addi	a2,s3,1440 # 1c0015a0 <__hal_debug_struct+0x1c>
1c00b6ae:	0585                	addi	a1,a1,1
1c00b6b0:	4501                	li	a0,0
1c00b6b2:	b5dff0ef          	jal	ra,1c00b20e <rt_periph_copy>
1c00b6b6:	3569                	jal	1c00b540 <__rt_io_unlock>
1c00b6b8:	000a2783          	lw	a5,0(s4)
1c00b6bc:	ef99                	bnez	a5,1c00b6da <__rt_io_uart_flush.constprop.10+0x92>
1c00b6be:	353d                	jal	1c00b4ec <__rt_io_lock>
1c00b6c0:	50fa                	lw	ra,188(sp)
1c00b6c2:	546a                	lw	s0,184(sp)
1c00b6c4:	54da                	lw	s1,180(sp)
1c00b6c6:	594a                	lw	s2,176(sp)
1c00b6c8:	59ba                	lw	s3,172(sp)
1c00b6ca:	5a2a                	lw	s4,168(sp)
1c00b6cc:	5a9a                	lw	s5,164(sp)
1c00b6ce:	6129                	addi	sp,sp,192
1c00b6d0:	8082                	ret
1c00b6d2:	35bd                	jal	1c00b540 <__rt_io_unlock>
1c00b6d4:	3bd5                	jal	1c00b4c8 <rt_event_execute.isra.2.constprop.11>
1c00b6d6:	3d19                	jal	1c00b4ec <__rt_io_lock>
1c00b6d8:	b751                	j	1c00b65c <__rt_io_uart_flush.constprop.10+0x14>
1c00b6da:	33fd                	jal	1c00b4c8 <rt_event_execute.isra.2.constprop.11>
1c00b6dc:	bff1                	j	1c00b6b8 <__rt_io_uart_flush.constprop.10+0x70>
1c00b6de:	0054                	addi	a3,sp,4
1c00b6e0:	8622                	mv	a2,s0
1c00b6e2:	5a098593          	addi	a1,s3,1440
1c00b6e6:	854a                	mv	a0,s2
1c00b6e8:	675000ef          	jal	ra,1c00c55c <rt_uart_cluster_write>
1c00b6ec:	4689                	li	a3,2
1c00b6ee:	00204737          	lui	a4,0x204
1c00b6f2:	09c14783          	lbu	a5,156(sp)
1c00b6f6:	0ff7f793          	andi	a5,a5,255
1c00b6fa:	c781                	beqz	a5,1c00b702 <__rt_io_uart_flush.constprop.10+0xba>
1c00b6fc:	000aac23          	sw	zero,24(s5)
1c00b700:	b7c1                	j	1c00b6c0 <__rt_io_uart_flush.constprop.10+0x78>
1c00b702:	c714                	sw	a3,8(a4)
1c00b704:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1eb054>
1c00b708:	c354                	sw	a3,4(a4)
1c00b70a:	b7e5                	j	1c00b6f2 <__rt_io_uart_flush.constprop.10+0xaa>

1c00b70c <memset>:
1c00b70c:	962a                	add	a2,a2,a0
1c00b70e:	87aa                	mv	a5,a0
1c00b710:	00c79363          	bne	a5,a2,1c00b716 <memset+0xa>
1c00b714:	8082                	ret
1c00b716:	00b780ab          	p.sb	a1,1(a5!)
1c00b71a:	bfdd                	j	1c00b710 <memset+0x4>

1c00b71c <memcpy>:
1c00b71c:	00a5e733          	or	a4,a1,a0
1c00b720:	fa273733          	p.bclr	a4,a4,29,2
1c00b724:	87aa                	mv	a5,a0
1c00b726:	c709                	beqz	a4,1c00b730 <memcpy+0x14>
1c00b728:	962e                	add	a2,a2,a1
1c00b72a:	00c59f63          	bne	a1,a2,1c00b748 <memcpy+0x2c>
1c00b72e:	8082                	ret
1c00b730:	fa263733          	p.bclr	a4,a2,29,2
1c00b734:	fb75                	bnez	a4,1c00b728 <memcpy+0xc>
1c00b736:	962e                	add	a2,a2,a1
1c00b738:	00c59363          	bne	a1,a2,1c00b73e <memcpy+0x22>
1c00b73c:	8082                	ret
1c00b73e:	0045a70b          	p.lw	a4,4(a1!)
1c00b742:	00e7a22b          	p.sw	a4,4(a5!)
1c00b746:	bfcd                	j	1c00b738 <memcpy+0x1c>
1c00b748:	0015c70b          	p.lbu	a4,1(a1!)
1c00b74c:	00e780ab          	p.sb	a4,1(a5!)
1c00b750:	bfe9                	j	1c00b72a <memcpy+0xe>

1c00b752 <memmove>:
1c00b752:	40b507b3          	sub	a5,a0,a1
1c00b756:	00c7e763          	bltu	a5,a2,1c00b764 <memmove+0x12>
1c00b75a:	962a                	add	a2,a2,a0
1c00b75c:	87aa                	mv	a5,a0
1c00b75e:	00c79f63          	bne	a5,a2,1c00b77c <memmove+0x2a>
1c00b762:	8082                	ret
1c00b764:	167d                	addi	a2,a2,-1
1c00b766:	00c507b3          	add	a5,a0,a2
1c00b76a:	95b2                	add	a1,a1,a2
1c00b76c:	0605                	addi	a2,a2,1
1c00b76e:	004640fb          	lp.setup	x1,a2,1c00b776 <memmove+0x24>
1c00b772:	fff5c70b          	p.lbu	a4,-1(a1!)
1c00b776:	fee78fab          	p.sb	a4,-1(a5!)
1c00b77a:	8082                	ret
1c00b77c:	0015c70b          	p.lbu	a4,1(a1!)
1c00b780:	00e780ab          	p.sb	a4,1(a5!)
1c00b784:	bfe9                	j	1c00b75e <memmove+0xc>

1c00b786 <strchr>:
1c00b786:	0ff5f593          	andi	a1,a1,255
1c00b78a:	00054703          	lbu	a4,0(a0)
1c00b78e:	87aa                	mv	a5,a0
1c00b790:	0505                	addi	a0,a0,1
1c00b792:	00b70563          	beq	a4,a1,1c00b79c <strchr+0x16>
1c00b796:	fb75                	bnez	a4,1c00b78a <strchr+0x4>
1c00b798:	c191                	beqz	a1,1c00b79c <strchr+0x16>
1c00b79a:	4781                	li	a5,0
1c00b79c:	853e                	mv	a0,a5
1c00b79e:	8082                	ret

1c00b7a0 <__rt_putc_debug_bridge>:
1c00b7a0:	1141                	addi	sp,sp,-16
1c00b7a2:	c422                	sw	s0,8(sp)
1c00b7a4:	1c001437          	lui	s0,0x1c001
1c00b7a8:	c226                	sw	s1,4(sp)
1c00b7aa:	c606                	sw	ra,12(sp)
1c00b7ac:	84aa                	mv	s1,a0
1c00b7ae:	58440413          	addi	s0,s0,1412 # 1c001584 <__hal_debug_struct>
1c00b7b2:	485c                	lw	a5,20(s0)
1c00b7b4:	c791                	beqz	a5,1c00b7c0 <__rt_putc_debug_bridge+0x20>
1c00b7b6:	06400513          	li	a0,100
1c00b7ba:	ca6fe0ef          	jal	ra,1c009c60 <rt_time_wait_us>
1c00b7be:	bfd5                	j	1c00b7b2 <__rt_putc_debug_bridge+0x12>
1c00b7c0:	4c1c                	lw	a5,24(s0)
1c00b7c2:	00178713          	addi	a4,a5,1
1c00b7c6:	97a2                	add	a5,a5,s0
1c00b7c8:	00978e23          	sb	s1,28(a5)
1c00b7cc:	cc18                	sw	a4,24(s0)
1c00b7ce:	4c14                	lw	a3,24(s0)
1c00b7d0:	08000793          	li	a5,128
1c00b7d4:	00f68463          	beq	a3,a5,1c00b7dc <__rt_putc_debug_bridge+0x3c>
1c00b7d8:	00a4b663          	p.bneimm	s1,10,1c00b7e4 <__rt_putc_debug_bridge+0x44>
1c00b7dc:	c701                	beqz	a4,1c00b7e4 <__rt_putc_debug_bridge+0x44>
1c00b7de:	c858                	sw	a4,20(s0)
1c00b7e0:	00042c23          	sw	zero,24(s0)
1c00b7e4:	4c1c                	lw	a5,24(s0)
1c00b7e6:	e799                	bnez	a5,1c00b7f4 <__rt_putc_debug_bridge+0x54>
1c00b7e8:	4422                	lw	s0,8(sp)
1c00b7ea:	40b2                	lw	ra,12(sp)
1c00b7ec:	4492                	lw	s1,4(sp)
1c00b7ee:	0141                	addi	sp,sp,16
1c00b7f0:	e84ff06f          	j	1c00ae74 <__rt_bridge_printf_flush>
1c00b7f4:	40b2                	lw	ra,12(sp)
1c00b7f6:	4422                	lw	s0,8(sp)
1c00b7f8:	4492                	lw	s1,4(sp)
1c00b7fa:	0141                	addi	sp,sp,16
1c00b7fc:	8082                	ret

1c00b7fe <__rt_putc_uart>:
1c00b7fe:	1101                	addi	sp,sp,-32
1c00b800:	c62a                	sw	a0,12(sp)
1c00b802:	ce06                	sw	ra,28(sp)
1c00b804:	3b41                	jal	1c00b594 <__rt_io_uart_wait_pending>
1c00b806:	1c0017b7          	lui	a5,0x1c001
1c00b80a:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00b80e:	4f94                	lw	a3,24(a5)
1c00b810:	4532                	lw	a0,12(sp)
1c00b812:	00168713          	addi	a4,a3,1
1c00b816:	cf98                	sw	a4,24(a5)
1c00b818:	97b6                	add	a5,a5,a3
1c00b81a:	00a78e23          	sb	a0,28(a5)
1c00b81e:	08000793          	li	a5,128
1c00b822:	00f70463          	beq	a4,a5,1c00b82a <__rt_putc_uart+0x2c>
1c00b826:	00a53563          	p.bneimm	a0,10,1c00b830 <__rt_putc_uart+0x32>
1c00b82a:	40f2                	lw	ra,28(sp)
1c00b82c:	6105                	addi	sp,sp,32
1c00b82e:	bd29                	j	1c00b648 <__rt_io_uart_flush.constprop.10>
1c00b830:	40f2                	lw	ra,28(sp)
1c00b832:	6105                	addi	sp,sp,32
1c00b834:	8082                	ret

1c00b836 <tfp_putc.isra.8>:
1c00b836:	1c0017b7          	lui	a5,0x1c001
1c00b83a:	7547a783          	lw	a5,1876(a5) # 1c001754 <_rt_io_uart>
1c00b83e:	c391                	beqz	a5,1c00b842 <tfp_putc.isra.8+0xc>
1c00b840:	bf7d                	j	1c00b7fe <__rt_putc_uart>
1c00b842:	1c0017b7          	lui	a5,0x1c001
1c00b846:	5947a783          	lw	a5,1428(a5) # 1c001594 <__hal_debug_struct+0x10>
1c00b84a:	c395                	beqz	a5,1c00b86e <tfp_putc.isra.8+0x38>
1c00b84c:	6689                	lui	a3,0x2
1c00b84e:	f14027f3          	csrr	a5,mhartid
1c00b852:	f8068693          	addi	a3,a3,-128 # 1f80 <__rt_hyper_pending_tasks_last+0x1a18>
1c00b856:	00379713          	slli	a4,a5,0x3
1c00b85a:	078a                	slli	a5,a5,0x2
1c00b85c:	ee873733          	p.bclr	a4,a4,23,8
1c00b860:	8ff5                	and	a5,a5,a3
1c00b862:	97ba                	add	a5,a5,a4
1c00b864:	1a120737          	lui	a4,0x1a120
1c00b868:	00a767a3          	p.sw	a0,a5(a4)
1c00b86c:	8082                	ret
1c00b86e:	bf0d                	j	1c00b7a0 <__rt_putc_debug_bridge>

1c00b870 <fputc_locked>:
1c00b870:	1141                	addi	sp,sp,-16
1c00b872:	c422                	sw	s0,8(sp)
1c00b874:	842a                	mv	s0,a0
1c00b876:	0ff57513          	andi	a0,a0,255
1c00b87a:	c606                	sw	ra,12(sp)
1c00b87c:	3f6d                	jal	1c00b836 <tfp_putc.isra.8>
1c00b87e:	8522                	mv	a0,s0
1c00b880:	40b2                	lw	ra,12(sp)
1c00b882:	4422                	lw	s0,8(sp)
1c00b884:	0141                	addi	sp,sp,16
1c00b886:	8082                	ret

1c00b888 <_prf_locked>:
1c00b888:	1101                	addi	sp,sp,-32
1c00b88a:	ce06                	sw	ra,28(sp)
1c00b88c:	c02a                	sw	a0,0(sp)
1c00b88e:	c62e                	sw	a1,12(sp)
1c00b890:	c432                	sw	a2,8(sp)
1c00b892:	c236                	sw	a3,4(sp)
1c00b894:	c59ff0ef          	jal	ra,1c00b4ec <__rt_io_lock>
1c00b898:	4692                	lw	a3,4(sp)
1c00b89a:	4622                	lw	a2,8(sp)
1c00b89c:	45b2                	lw	a1,12(sp)
1c00b89e:	4502                	lw	a0,0(sp)
1c00b8a0:	22c5                	jal	1c00ba80 <_prf>
1c00b8a2:	c02a                	sw	a0,0(sp)
1c00b8a4:	3971                	jal	1c00b540 <__rt_io_unlock>
1c00b8a6:	40f2                	lw	ra,28(sp)
1c00b8a8:	4502                	lw	a0,0(sp)
1c00b8aa:	6105                	addi	sp,sp,32
1c00b8ac:	8082                	ret

1c00b8ae <exit>:
1c00b8ae:	1141                	addi	sp,sp,-16
1c00b8b0:	c422                	sw	s0,8(sp)
1c00b8b2:	1a104437          	lui	s0,0x1a104
1c00b8b6:	0a040793          	addi	a5,s0,160 # 1a1040a0 <__l1_end+0xa0fd088>
1c00b8ba:	c606                	sw	ra,12(sp)
1c00b8bc:	c226                	sw	s1,4(sp)
1c00b8be:	c04a                	sw	s2,0(sp)
1c00b8c0:	1c0014b7          	lui	s1,0x1c001
1c00b8c4:	c1f54933          	p.bset	s2,a0,0,31
1c00b8c8:	0127a023          	sw	s2,0(a5)
1c00b8cc:	58448493          	addi	s1,s1,1412 # 1c001584 <__hal_debug_struct>
1c00b8d0:	da4ff0ef          	jal	ra,1c00ae74 <__rt_bridge_printf_flush>
1c00b8d4:	0124a623          	sw	s2,12(s1)
1c00b8d8:	d5eff0ef          	jal	ra,1c00ae36 <__rt_bridge_send_notif>
1c00b8dc:	449c                	lw	a5,8(s1)
1c00b8de:	cb91                	beqz	a5,1c00b8f2 <exit+0x44>
1c00b8e0:	07440413          	addi	s0,s0,116
1c00b8e4:	401c                	lw	a5,0(s0)
1c00b8e6:	cc9797b3          	p.extractu	a5,a5,6,9
1c00b8ea:	fe77bde3          	p.bneimm	a5,7,1c00b8e4 <exit+0x36>
1c00b8ee:	d68ff0ef          	jal	ra,1c00ae56 <__rt_bridge_clear_notif>
1c00b8f2:	a001                	j	1c00b8f2 <exit+0x44>

1c00b8f4 <abort>:
1c00b8f4:	1141                	addi	sp,sp,-16
1c00b8f6:	557d                	li	a0,-1
1c00b8f8:	c606                	sw	ra,12(sp)
1c00b8fa:	3f55                	jal	1c00b8ae <exit>

1c00b8fc <__rt_io_init>:
1c00b8fc:	1c0017b7          	lui	a5,0x1c001
1c00b900:	57478793          	addi	a5,a5,1396 # 1c001574 <__rt_io_fc_lock>
1c00b904:	0007a223          	sw	zero,4(a5)
1c00b908:	0007a023          	sw	zero,0(a5)
1c00b90c:	0007a623          	sw	zero,12(a5)
1c00b910:	1c0017b7          	lui	a5,0x1c001
1c00b914:	7407aa23          	sw	zero,1876(a5) # 1c001754 <_rt_io_uart>
1c00b918:	1c0017b7          	lui	a5,0x1c001
1c00b91c:	7407a623          	sw	zero,1868(a5) # 1c00174c <__rt_io_event_current>
1c00b920:	1c0017b7          	lui	a5,0x1c001
1c00b924:	7587a783          	lw	a5,1880(a5) # 1c001758 <__rt_iodev>
1c00b928:	0217be63          	p.bneimm	a5,1,1c00b964 <__rt_io_init+0x68>
1c00b92c:	1c00b5b7          	lui	a1,0x1c00b
1c00b930:	1141                	addi	sp,sp,-16
1c00b932:	4601                	li	a2,0
1c00b934:	46a58593          	addi	a1,a1,1130 # 1c00b46a <__rt_io_start>
1c00b938:	4501                	li	a0,0
1c00b93a:	c606                	sw	ra,12(sp)
1c00b93c:	a2eff0ef          	jal	ra,1c00ab6a <__rt_cbsys_add>
1c00b940:	1c00b5b7          	lui	a1,0x1c00b
1c00b944:	62258593          	addi	a1,a1,1570 # 1c00b622 <__rt_io_stop>
1c00b948:	4601                	li	a2,0
1c00b94a:	4505                	li	a0,1
1c00b94c:	a1eff0ef          	jal	ra,1c00ab6a <__rt_cbsys_add>
1c00b950:	40b2                	lw	ra,12(sp)
1c00b952:	1c0017b7          	lui	a5,0x1c001
1c00b956:	7407a823          	sw	zero,1872(a5) # 1c001750 <__rt_io_pending_flush>
1c00b95a:	4585                	li	a1,1
1c00b95c:	4501                	li	a0,0
1c00b95e:	0141                	addi	sp,sp,16
1c00b960:	d13fd06f          	j	1c009672 <rt_event_alloc>
1c00b964:	8082                	ret

1c00b966 <printf>:
1c00b966:	7139                	addi	sp,sp,-64
1c00b968:	d432                	sw	a2,40(sp)
1c00b96a:	862a                	mv	a2,a0
1c00b96c:	1c00c537          	lui	a0,0x1c00c
1c00b970:	d22e                	sw	a1,36(sp)
1c00b972:	d636                	sw	a3,44(sp)
1c00b974:	4589                	li	a1,2
1c00b976:	1054                	addi	a3,sp,36
1c00b978:	87050513          	addi	a0,a0,-1936 # 1c00b870 <fputc_locked>
1c00b97c:	ce06                	sw	ra,28(sp)
1c00b97e:	d83a                	sw	a4,48(sp)
1c00b980:	da3e                	sw	a5,52(sp)
1c00b982:	dc42                	sw	a6,56(sp)
1c00b984:	de46                	sw	a7,60(sp)
1c00b986:	c636                	sw	a3,12(sp)
1c00b988:	3701                	jal	1c00b888 <_prf_locked>
1c00b98a:	40f2                	lw	ra,28(sp)
1c00b98c:	6121                	addi	sp,sp,64
1c00b98e:	8082                	ret

1c00b990 <_to_x>:
1c00b990:	872a                	mv	a4,a0
1c00b992:	87aa                	mv	a5,a0
1c00b994:	4325                	li	t1,9
1c00b996:	02c5f8b3          	remu	a7,a1,a2
1c00b99a:	02700513          	li	a0,39
1c00b99e:	02c5d5b3          	divu	a1,a1,a2
1c00b9a2:	0ff8f813          	andi	a6,a7,255
1c00b9a6:	01136363          	bltu	t1,a7,1c00b9ac <_to_x+0x1c>
1c00b9aa:	4501                	li	a0,0
1c00b9ac:	03080813          	addi	a6,a6,48
1c00b9b0:	9542                	add	a0,a0,a6
1c00b9b2:	00a780ab          	p.sb	a0,1(a5!)
1c00b9b6:	f1e5                	bnez	a1,1c00b996 <_to_x+0x6>
1c00b9b8:	03000613          	li	a2,48
1c00b9bc:	40e78533          	sub	a0,a5,a4
1c00b9c0:	00d54763          	blt	a0,a3,1c00b9ce <_to_x+0x3e>
1c00b9c4:	fe078fab          	p.sb	zero,-1(a5!)
1c00b9c8:	00f76663          	bltu	a4,a5,1c00b9d4 <_to_x+0x44>
1c00b9cc:	8082                	ret
1c00b9ce:	00c780ab          	p.sb	a2,1(a5!)
1c00b9d2:	b7ed                	j	1c00b9bc <_to_x+0x2c>
1c00b9d4:	00074603          	lbu	a2,0(a4) # 1a120000 <__l1_end+0xa118fe8>
1c00b9d8:	0007c683          	lbu	a3,0(a5)
1c00b9dc:	fec78fab          	p.sb	a2,-1(a5!)
1c00b9e0:	00d700ab          	p.sb	a3,1(a4!)
1c00b9e4:	b7d5                	j	1c00b9c8 <_to_x+0x38>

1c00b9e6 <_rlrshift>:
1c00b9e6:	411c                	lw	a5,0(a0)
1c00b9e8:	4154                	lw	a3,4(a0)
1c00b9ea:	fc17b733          	p.bclr	a4,a5,30,1
1c00b9ee:	01f69613          	slli	a2,a3,0x1f
1c00b9f2:	8385                	srli	a5,a5,0x1
1c00b9f4:	8fd1                	or	a5,a5,a2
1c00b9f6:	97ba                	add	a5,a5,a4
1c00b9f8:	8285                	srli	a3,a3,0x1
1c00b9fa:	00e7b733          	sltu	a4,a5,a4
1c00b9fe:	9736                	add	a4,a4,a3
1c00ba00:	c11c                	sw	a5,0(a0)
1c00ba02:	c158                	sw	a4,4(a0)
1c00ba04:	8082                	ret

1c00ba06 <_ldiv5>:
1c00ba06:	4118                	lw	a4,0(a0)
1c00ba08:	4154                	lw	a3,4(a0)
1c00ba0a:	4615                	li	a2,5
1c00ba0c:	00270793          	addi	a5,a4,2
1c00ba10:	00e7b733          	sltu	a4,a5,a4
1c00ba14:	9736                	add	a4,a4,a3
1c00ba16:	02c755b3          	divu	a1,a4,a2
1c00ba1a:	42b61733          	p.msu	a4,a2,a1
1c00ba1e:	01d71693          	slli	a3,a4,0x1d
1c00ba22:	0037d713          	srli	a4,a5,0x3
1c00ba26:	8f55                	or	a4,a4,a3
1c00ba28:	02c75733          	divu	a4,a4,a2
1c00ba2c:	01d75693          	srli	a3,a4,0x1d
1c00ba30:	070e                	slli	a4,a4,0x3
1c00ba32:	42e617b3          	p.msu	a5,a2,a4
1c00ba36:	95b6                	add	a1,a1,a3
1c00ba38:	02c7d7b3          	divu	a5,a5,a2
1c00ba3c:	973e                	add	a4,a4,a5
1c00ba3e:	00f737b3          	sltu	a5,a4,a5
1c00ba42:	97ae                	add	a5,a5,a1
1c00ba44:	c118                	sw	a4,0(a0)
1c00ba46:	c15c                	sw	a5,4(a0)
1c00ba48:	8082                	ret

1c00ba4a <_get_digit>:
1c00ba4a:	419c                	lw	a5,0(a1)
1c00ba4c:	03000713          	li	a4,48
1c00ba50:	02f05563          	blez	a5,1c00ba7a <_get_digit+0x30>
1c00ba54:	17fd                	addi	a5,a5,-1
1c00ba56:	c19c                	sw	a5,0(a1)
1c00ba58:	411c                	lw	a5,0(a0)
1c00ba5a:	4729                	li	a4,10
1c00ba5c:	4150                	lw	a2,4(a0)
1c00ba5e:	02f706b3          	mul	a3,a4,a5
1c00ba62:	02f737b3          	mulhu	a5,a4,a5
1c00ba66:	c114                	sw	a3,0(a0)
1c00ba68:	42c707b3          	p.mac	a5,a4,a2
1c00ba6c:	01c7d713          	srli	a4,a5,0x1c
1c00ba70:	c7c7b7b3          	p.bclr	a5,a5,3,28
1c00ba74:	03070713          	addi	a4,a4,48
1c00ba78:	c15c                	sw	a5,4(a0)
1c00ba7a:	0ff77513          	andi	a0,a4,255
1c00ba7e:	8082                	ret

1c00ba80 <_prf>:
1c00ba80:	714d                	addi	sp,sp,-336
1c00ba82:	14912223          	sw	s1,324(sp)
1c00ba86:	15212023          	sw	s2,320(sp)
1c00ba8a:	13812423          	sw	s8,296(sp)
1c00ba8e:	14112623          	sw	ra,332(sp)
1c00ba92:	14812423          	sw	s0,328(sp)
1c00ba96:	13312e23          	sw	s3,316(sp)
1c00ba9a:	13412c23          	sw	s4,312(sp)
1c00ba9e:	13512a23          	sw	s5,308(sp)
1c00baa2:	13612823          	sw	s6,304(sp)
1c00baa6:	13712623          	sw	s7,300(sp)
1c00baaa:	13912223          	sw	s9,292(sp)
1c00baae:	13a12023          	sw	s10,288(sp)
1c00bab2:	11b12e23          	sw	s11,284(sp)
1c00bab6:	cc2a                	sw	a0,24(sp)
1c00bab8:	ce2e                	sw	a1,28(sp)
1c00baba:	84b2                	mv	s1,a2
1c00babc:	8c36                	mv	s8,a3
1c00babe:	4901                	li	s2,0
1c00bac0:	0004c503          	lbu	a0,0(s1)
1c00bac4:	00148b93          	addi	s7,s1,1
1c00bac8:	c919                	beqz	a0,1c00bade <_prf+0x5e>
1c00baca:	02500793          	li	a5,37
1c00bace:	14f50763          	beq	a0,a5,1c00bc1c <_prf+0x19c>
1c00bad2:	45f2                	lw	a1,28(sp)
1c00bad4:	4762                	lw	a4,24(sp)
1c00bad6:	9702                	jalr	a4
1c00bad8:	05f53063          	p.bneimm	a0,-1,1c00bb18 <_prf+0x98>
1c00badc:	597d                	li	s2,-1
1c00bade:	14c12083          	lw	ra,332(sp)
1c00bae2:	14812403          	lw	s0,328(sp)
1c00bae6:	854a                	mv	a0,s2
1c00bae8:	14412483          	lw	s1,324(sp)
1c00baec:	14012903          	lw	s2,320(sp)
1c00baf0:	13c12983          	lw	s3,316(sp)
1c00baf4:	13812a03          	lw	s4,312(sp)
1c00baf8:	13412a83          	lw	s5,308(sp)
1c00bafc:	13012b03          	lw	s6,304(sp)
1c00bb00:	12c12b83          	lw	s7,300(sp)
1c00bb04:	12812c03          	lw	s8,296(sp)
1c00bb08:	12412c83          	lw	s9,292(sp)
1c00bb0c:	12012d03          	lw	s10,288(sp)
1c00bb10:	11c12d83          	lw	s11,284(sp)
1c00bb14:	6171                	addi	sp,sp,336
1c00bb16:	8082                	ret
1c00bb18:	0905                	addi	s2,s2,1
1c00bb1a:	8a62                	mv	s4,s8
1c00bb1c:	84de                	mv	s1,s7
1c00bb1e:	8c52                	mv	s8,s4
1c00bb20:	b745                	j	1c00bac0 <_prf+0x40>
1c00bb22:	0f3a8463          	beq	s5,s3,1c00bc0a <_prf+0x18a>
1c00bb26:	0d59e763          	bltu	s3,s5,1c00bbf4 <_prf+0x174>
1c00bb2a:	fa0a8ae3          	beqz	s5,1c00bade <_prf+0x5e>
1c00bb2e:	0dba8c63          	beq	s5,s11,1c00bc06 <_prf+0x186>
1c00bb32:	8ba6                	mv	s7,s1
1c00bb34:	000bca83          	lbu	s5,0(s7)
1c00bb38:	1c0017b7          	lui	a5,0x1c001
1c00bb3c:	b8478513          	addi	a0,a5,-1148 # 1c000b84 <PIo2+0x228>
1c00bb40:	85d6                	mv	a1,s5
1c00bb42:	001b8493          	addi	s1,s7,1
1c00bb46:	c41ff0ef          	jal	ra,1c00b786 <strchr>
1c00bb4a:	fd61                	bnez	a0,1c00bb22 <_prf+0xa2>
1c00bb4c:	02a00693          	li	a3,42
1c00bb50:	0eda9863          	bne	s5,a3,1c00bc40 <_prf+0x1c0>
1c00bb54:	000c2983          	lw	s3,0(s8)
1c00bb58:	004c0693          	addi	a3,s8,4
1c00bb5c:	0009d663          	bgez	s3,1c00bb68 <_prf+0xe8>
1c00bb60:	4705                	li	a4,1
1c00bb62:	413009b3          	neg	s3,s3
1c00bb66:	ca3a                	sw	a4,20(sp)
1c00bb68:	0004ca83          	lbu	s5,0(s1)
1c00bb6c:	8c36                	mv	s8,a3
1c00bb6e:	002b8493          	addi	s1,s7,2
1c00bb72:	0c800713          	li	a4,200
1c00bb76:	02e00693          	li	a3,46
1c00bb7a:	04e9d9b3          	p.minu	s3,s3,a4
1c00bb7e:	5d7d                	li	s10,-1
1c00bb80:	02da9463          	bne	s5,a3,1c00bba8 <_prf+0x128>
1c00bb84:	0004c703          	lbu	a4,0(s1)
1c00bb88:	02a00793          	li	a5,42
1c00bb8c:	0ef71d63          	bne	a4,a5,1c00bc86 <_prf+0x206>
1c00bb90:	000c2d03          	lw	s10,0(s8)
1c00bb94:	0485                	addi	s1,s1,1
1c00bb96:	0c11                	addi	s8,s8,4
1c00bb98:	0c800793          	li	a5,200
1c00bb9c:	01a7d363          	ble	s10,a5,1c00bba2 <_prf+0x122>
1c00bba0:	5d7d                	li	s10,-1
1c00bba2:	0004ca83          	lbu	s5,0(s1)
1c00bba6:	0485                	addi	s1,s1,1
1c00bba8:	1c0017b7          	lui	a5,0x1c001
1c00bbac:	85d6                	mv	a1,s5
1c00bbae:	b8c78513          	addi	a0,a5,-1140 # 1c000b8c <PIo2+0x230>
1c00bbb2:	bd5ff0ef          	jal	ra,1c00b786 <strchr>
1c00bbb6:	c501                	beqz	a0,1c00bbbe <_prf+0x13e>
1c00bbb8:	0004ca83          	lbu	s5,0(s1)
1c00bbbc:	0485                	addi	s1,s1,1
1c00bbbe:	06700693          	li	a3,103
1c00bbc2:	1356c563          	blt	a3,s5,1c00bcec <_prf+0x26c>
1c00bbc6:	06500693          	li	a3,101
1c00bbca:	20dad163          	ble	a3,s5,1c00bdcc <_prf+0x34c>
1c00bbce:	04700693          	li	a3,71
1c00bbd2:	0b56ce63          	blt	a3,s5,1c00bc8e <_prf+0x20e>
1c00bbd6:	04500693          	li	a3,69
1c00bbda:	1edad963          	ble	a3,s5,1c00bdcc <_prf+0x34c>
1c00bbde:	f00a80e3          	beqz	s5,1c00bade <_prf+0x5e>
1c00bbe2:	02500713          	li	a4,37
1c00bbe6:	64ea8d63          	beq	s5,a4,1c00c240 <_prf+0x7c0>
1c00bbea:	0c800713          	li	a4,200
1c00bbee:	67575163          	ble	s5,a4,1c00c250 <_prf+0x7d0>
1c00bbf2:	b5ed                	j	1c00badc <_prf+0x5c>
1c00bbf4:	034a8163          	beq	s5,s4,1c00bc16 <_prf+0x196>
1c00bbf8:	016a8b63          	beq	s5,s6,1c00bc0e <_prf+0x18e>
1c00bbfc:	f3aa9be3          	bne	s5,s10,1c00bb32 <_prf+0xb2>
1c00bc00:	4785                	li	a5,1
1c00bc02:	c83e                	sw	a5,16(sp)
1c00bc04:	b73d                	j	1c00bb32 <_prf+0xb2>
1c00bc06:	4405                	li	s0,1
1c00bc08:	b72d                	j	1c00bb32 <_prf+0xb2>
1c00bc0a:	4c85                	li	s9,1
1c00bc0c:	b71d                	j	1c00bb32 <_prf+0xb2>
1c00bc0e:	03000713          	li	a4,48
1c00bc12:	c63a                	sw	a4,12(sp)
1c00bc14:	bf39                	j	1c00bb32 <_prf+0xb2>
1c00bc16:	4785                	li	a5,1
1c00bc18:	ca3e                	sw	a5,20(sp)
1c00bc1a:	bf21                	j	1c00bb32 <_prf+0xb2>
1c00bc1c:	02000713          	li	a4,32
1c00bc20:	c63a                	sw	a4,12(sp)
1c00bc22:	4401                	li	s0,0
1c00bc24:	c802                	sw	zero,16(sp)
1c00bc26:	ca02                	sw	zero,20(sp)
1c00bc28:	4c81                	li	s9,0
1c00bc2a:	02300993          	li	s3,35
1c00bc2e:	02d00a13          	li	s4,45
1c00bc32:	03000b13          	li	s6,48
1c00bc36:	02b00d13          	li	s10,43
1c00bc3a:	02000d93          	li	s11,32
1c00bc3e:	bddd                	j	1c00bb34 <_prf+0xb4>
1c00bc40:	fd0a8693          	addi	a3,s5,-48
1c00bc44:	4625                	li	a2,9
1c00bc46:	4981                	li	s3,0
1c00bc48:	f2d665e3          	bltu	a2,a3,1c00bb72 <_prf+0xf2>
1c00bc4c:	46a5                	li	a3,9
1c00bc4e:	45a9                	li	a1,10
1c00bc50:	84de                	mv	s1,s7
1c00bc52:	0014c70b          	p.lbu	a4,1(s1!)
1c00bc56:	fd070613          	addi	a2,a4,-48
1c00bc5a:	8aba                	mv	s5,a4
1c00bc5c:	f0c6ebe3          	bltu	a3,a2,1c00bb72 <_prf+0xf2>
1c00bc60:	42b98733          	p.mac	a4,s3,a1
1c00bc64:	8ba6                	mv	s7,s1
1c00bc66:	fd070993          	addi	s3,a4,-48
1c00bc6a:	b7dd                	j	1c00bc50 <_prf+0x1d0>
1c00bc6c:	42ad07b3          	p.mac	a5,s10,a0
1c00bc70:	84b6                	mv	s1,a3
1c00bc72:	fd078d13          	addi	s10,a5,-48
1c00bc76:	86a6                	mv	a3,s1
1c00bc78:	0016c78b          	p.lbu	a5,1(a3!)
1c00bc7c:	fd078593          	addi	a1,a5,-48
1c00bc80:	feb676e3          	bleu	a1,a2,1c00bc6c <_prf+0x1ec>
1c00bc84:	bf11                	j	1c00bb98 <_prf+0x118>
1c00bc86:	4d01                	li	s10,0
1c00bc88:	4625                	li	a2,9
1c00bc8a:	4529                	li	a0,10
1c00bc8c:	b7ed                	j	1c00bc76 <_prf+0x1f6>
1c00bc8e:	06300693          	li	a3,99
1c00bc92:	0cda8e63          	beq	s5,a3,1c00bd6e <_prf+0x2ee>
1c00bc96:	0756cb63          	blt	a3,s5,1c00bd0c <_prf+0x28c>
1c00bc9a:	05800693          	li	a3,88
1c00bc9e:	f4da96e3          	bne	s5,a3,1c00bbea <_prf+0x16a>
1c00bca2:	04410b93          	addi	s7,sp,68
1c00bca6:	004c0a13          	addi	s4,s8,4
1c00bcaa:	000c2583          	lw	a1,0(s8)
1c00bcae:	845e                	mv	s0,s7
1c00bcb0:	000c8963          	beqz	s9,1c00bcc2 <_prf+0x242>
1c00bcb4:	76e1                	lui	a3,0xffff8
1c00bcb6:	8306c693          	xori	a3,a3,-2000
1c00bcba:	04d11223          	sh	a3,68(sp)
1c00bcbe:	04610413          	addi	s0,sp,70
1c00bcc2:	86ea                	mv	a3,s10
1c00bcc4:	4641                	li	a2,16
1c00bcc6:	8522                	mv	a0,s0
1c00bcc8:	31e1                	jal	1c00b990 <_to_x>
1c00bcca:	05800693          	li	a3,88
1c00bcce:	00da9863          	bne	s5,a3,1c00bcde <_prf+0x25e>
1c00bcd2:	86de                	mv	a3,s7
1c00bcd4:	45e5                	li	a1,25
1c00bcd6:	0016c78b          	p.lbu	a5,1(a3!) # ffff8001 <pulp__FC+0xffff8002>
1c00bcda:	54079963          	bnez	a5,1c00c22c <_prf+0x7ac>
1c00bcde:	41740433          	sub	s0,s0,s7
1c00bce2:	9522                	add	a0,a0,s0
1c00bce4:	01903433          	snez	s0,s9
1c00bce8:	0406                	slli	s0,s0,0x1
1c00bcea:	a0f1                	j	1c00bdb6 <_prf+0x336>
1c00bcec:	07000693          	li	a3,112
1c00bcf0:	4eda8763          	beq	s5,a3,1c00c1de <_prf+0x75e>
1c00bcf4:	0556c163          	blt	a3,s5,1c00bd36 <_prf+0x2b6>
1c00bcf8:	06e00693          	li	a3,110
1c00bcfc:	46da8463          	beq	s5,a3,1c00c164 <_prf+0x6e4>
1c00bd00:	4756c963          	blt	a3,s5,1c00c172 <_prf+0x6f2>
1c00bd04:	06900693          	li	a3,105
1c00bd08:	eeda91e3          	bne	s5,a3,1c00bbea <_prf+0x16a>
1c00bd0c:	000c2a83          	lw	s5,0(s8)
1c00bd10:	004c0a13          	addi	s4,s8,4
1c00bd14:	04410b13          	addi	s6,sp,68
1c00bd18:	060ad663          	bgez	s5,1c00bd84 <_prf+0x304>
1c00bd1c:	02d00693          	li	a3,45
1c00bd20:	04d10223          	sb	a3,68(sp)
1c00bd24:	80000737          	lui	a4,0x80000
1c00bd28:	415005b3          	neg	a1,s5
1c00bd2c:	06ea9663          	bne	s5,a4,1c00bd98 <_prf+0x318>
1c00bd30:	800005b7          	lui	a1,0x80000
1c00bd34:	a095                	j	1c00bd98 <_prf+0x318>
1c00bd36:	07500693          	li	a3,117
1c00bd3a:	4cda8f63          	beq	s5,a3,1c00c218 <_prf+0x798>
1c00bd3e:	07800693          	li	a3,120
1c00bd42:	f6da80e3          	beq	s5,a3,1c00bca2 <_prf+0x222>
1c00bd46:	07300693          	li	a3,115
1c00bd4a:	eada90e3          	bne	s5,a3,1c00bbea <_prf+0x16a>
1c00bd4e:	000c2583          	lw	a1,0(s8)
1c00bd52:	004c0a13          	addi	s4,s8,4
1c00bd56:	4c81                	li	s9,0
1c00bd58:	86ae                	mv	a3,a1
1c00bd5a:	0c8350fb          	lp.setupi	x1,200,1c00bd66 <_prf+0x2e6>
1c00bd5e:	0016c60b          	p.lbu	a2,1(a3!)
1c00bd62:	4a060063          	beqz	a2,1c00c202 <_prf+0x782>
1c00bd66:	0c85                	addi	s9,s9,1
1c00bd68:	480d5f63          	bgez	s10,1c00c206 <_prf+0x786>
1c00bd6c:	a14d                	j	1c00c20e <_prf+0x78e>
1c00bd6e:	000c2783          	lw	a5,0(s8)
1c00bd72:	004c0a13          	addi	s4,s8,4
1c00bd76:	040102a3          	sb	zero,69(sp)
1c00bd7a:	04f10223          	sb	a5,68(sp)
1c00bd7e:	4c85                	li	s9,1
1c00bd80:	4401                	li	s0,0
1c00bd82:	a919                	j	1c00c198 <_prf+0x718>
1c00bd84:	47c2                	lw	a5,16(sp)
1c00bd86:	02b00693          	li	a3,43
1c00bd8a:	e781                	bnez	a5,1c00bd92 <_prf+0x312>
1c00bd8c:	c81d                	beqz	s0,1c00bdc2 <_prf+0x342>
1c00bd8e:	02000693          	li	a3,32
1c00bd92:	04d10223          	sb	a3,68(sp)
1c00bd96:	85d6                	mv	a1,s5
1c00bd98:	04510c13          	addi	s8,sp,69
1c00bd9c:	86ea                	mv	a3,s10
1c00bd9e:	4629                	li	a2,10
1c00bda0:	8562                	mv	a0,s8
1c00bda2:	befff0ef          	jal	ra,1c00b990 <_to_x>
1c00bda6:	4742                	lw	a4,16(sp)
1c00bda8:	9562                	add	a0,a0,s8
1c00bdaa:	41650533          	sub	a0,a0,s6
1c00bdae:	ef09                	bnez	a4,1c00bdc8 <_prf+0x348>
1c00bdb0:	e019                	bnez	s0,1c00bdb6 <_prf+0x336>
1c00bdb2:	01fad413          	srli	s0,s5,0x1f
1c00bdb6:	0bfd2363          	p.beqimm	s10,-1,1c00be5c <_prf+0x3dc>
1c00bdba:	02000713          	li	a4,32
1c00bdbe:	c63a                	sw	a4,12(sp)
1c00bdc0:	a871                	j	1c00be5c <_prf+0x3dc>
1c00bdc2:	85d6                	mv	a1,s5
1c00bdc4:	8c5a                	mv	s8,s6
1c00bdc6:	bfd9                	j	1c00bd9c <_prf+0x31c>
1c00bdc8:	4442                	lw	s0,16(sp)
1c00bdca:	b7f5                	j	1c00bdb6 <_prf+0x336>
1c00bdcc:	0c1d                	addi	s8,s8,7
1c00bdce:	c40c3c33          	p.bclr	s8,s8,2,0
1c00bdd2:	000c2883          	lw	a7,0(s8)
1c00bdd6:	004c2303          	lw	t1,4(s8)
1c00bdda:	800007b7          	lui	a5,0x80000
1c00bdde:	0158d593          	srli	a1,a7,0x15
1c00bde2:	00b31693          	slli	a3,t1,0xb
1c00bde6:	8ecd                	or	a3,a3,a1
1c00bde8:	fff7c793          	not	a5,a5
1c00bdec:	01435613          	srli	a2,t1,0x14
1c00bdf0:	08ae                	slli	a7,a7,0xb
1c00bdf2:	8efd                	and	a3,a3,a5
1c00bdf4:	e8b63633          	p.bclr	a2,a2,20,11
1c00bdf8:	d846                	sw	a7,48(sp)
1c00bdfa:	da36                	sw	a3,52(sp)
1c00bdfc:	7ff00593          	li	a1,2047
1c00be00:	008c0a13          	addi	s4,s8,8
1c00be04:	08b61d63          	bne	a2,a1,1c00be9e <_prf+0x41e>
1c00be08:	00d0                	addi	a2,sp,68
1c00be0a:	8732                	mv	a4,a2
1c00be0c:	00035863          	bgez	t1,1c00be1c <_prf+0x39c>
1c00be10:	02d00713          	li	a4,45
1c00be14:	04e10223          	sb	a4,68(sp)
1c00be18:	04510713          	addi	a4,sp,69
1c00be1c:	00d8e6b3          	or	a3,a7,a3
1c00be20:	fbfa8793          	addi	a5,s5,-65
1c00be24:	00370513          	addi	a0,a4,3 # 80000003 <pulp__FC+0x80000004>
1c00be28:	eaa1                	bnez	a3,1c00be78 <_prf+0x3f8>
1c00be2a:	46e5                	li	a3,25
1c00be2c:	02f6ee63          	bltu	a3,a5,1c00be68 <_prf+0x3e8>
1c00be30:	6795                	lui	a5,0x5
1c00be32:	e4978793          	addi	a5,a5,-439 # 4e49 <__rt_hyper_pending_tasks_last+0x48e1>
1c00be36:	00f71023          	sh	a5,0(a4)
1c00be3a:	04600793          	li	a5,70
1c00be3e:	00f70123          	sb	a5,2(a4)
1c00be42:	000701a3          	sb	zero,3(a4)
1c00be46:	8d11                	sub	a0,a0,a2
1c00be48:	47c2                	lw	a5,16(sp)
1c00be4a:	46079163          	bnez	a5,1c00c2ac <_prf+0x82c>
1c00be4e:	e419                	bnez	s0,1c00be5c <_prf+0x3dc>
1c00be50:	04414403          	lbu	s0,68(sp)
1c00be54:	fd340413          	addi	s0,s0,-45
1c00be58:	00143413          	seqz	s0,s0
1c00be5c:	0c800793          	li	a5,200
1c00be60:	c6a7cee3          	blt	a5,a0,1c00badc <_prf+0x5c>
1c00be64:	8caa                	mv	s9,a0
1c00be66:	ae0d                	j	1c00c198 <_prf+0x718>
1c00be68:	679d                	lui	a5,0x7
1c00be6a:	e6978793          	addi	a5,a5,-407 # 6e69 <__rt_hyper_pending_tasks_last+0x6901>
1c00be6e:	00f71023          	sh	a5,0(a4)
1c00be72:	06600793          	li	a5,102
1c00be76:	b7e1                	j	1c00be3e <_prf+0x3be>
1c00be78:	46e5                	li	a3,25
1c00be7a:	00f6ea63          	bltu	a3,a5,1c00be8e <_prf+0x40e>
1c00be7e:	6791                	lui	a5,0x4
1c00be80:	14e78793          	addi	a5,a5,334 # 414e <__rt_hyper_pending_tasks_last+0x3be6>
1c00be84:	00f71023          	sh	a5,0(a4)
1c00be88:	04e00793          	li	a5,78
1c00be8c:	bf4d                	j	1c00be3e <_prf+0x3be>
1c00be8e:	6799                	lui	a5,0x6
1c00be90:	16e78793          	addi	a5,a5,366 # 616e <__rt_hyper_pending_tasks_last+0x5c06>
1c00be94:	00f71023          	sh	a5,0(a4)
1c00be98:	06e00793          	li	a5,110
1c00be9c:	b74d                	j	1c00be3e <_prf+0x3be>
1c00be9e:	04600593          	li	a1,70
1c00bea2:	00ba9463          	bne	s5,a1,1c00beaa <_prf+0x42a>
1c00bea6:	06600a93          	li	s5,102
1c00beaa:	011665b3          	or	a1,a2,a7
1c00beae:	8dd5                	or	a1,a1,a3
1c00beb0:	c5d9                	beqz	a1,1c00bf3e <_prf+0x4be>
1c00beb2:	80000737          	lui	a4,0x80000
1c00beb6:	8ed9                	or	a3,a3,a4
1c00beb8:	da36                	sw	a3,52(sp)
1c00beba:	d846                	sw	a7,48(sp)
1c00bebc:	c0260c13          	addi	s8,a2,-1022
1c00bec0:	02d00693          	li	a3,45
1c00bec4:	00034b63          	bltz	t1,1c00beda <_prf+0x45a>
1c00bec8:	47c2                	lw	a5,16(sp)
1c00beca:	02b00693          	li	a3,43
1c00bece:	e791                	bnez	a5,1c00beda <_prf+0x45a>
1c00bed0:	04410b13          	addi	s6,sp,68
1c00bed4:	c419                	beqz	s0,1c00bee2 <_prf+0x462>
1c00bed6:	02000693          	li	a3,32
1c00beda:	04d10223          	sb	a3,68(sp)
1c00bede:	04510b13          	addi	s6,sp,69
1c00bee2:	4b81                	li	s7,0
1c00bee4:	55f9                	li	a1,-2
1c00bee6:	06bc4163          	blt	s8,a1,1c00bf48 <_prf+0x4c8>
1c00beea:	0b804763          	bgtz	s8,1c00bf98 <_prf+0x518>
1c00beee:	1808                	addi	a0,sp,48
1c00bef0:	0c05                	addi	s8,s8,1
1c00bef2:	af5ff0ef          	jal	ra,1c00b9e6 <_rlrshift>
1c00bef6:	fe4c3ce3          	p.bneimm	s8,4,1c00beee <_prf+0x46e>
1c00befa:	000d5363          	bgez	s10,1c00bf00 <_prf+0x480>
1c00befe:	4d19                	li	s10,6
1c00bf00:	c05ab5b3          	p.bclr	a1,s5,0,5
1c00bf04:	04700513          	li	a0,71
1c00bf08:	0ca59463          	bne	a1,a0,1c00bfd0 <_prf+0x550>
1c00bf0c:	4c01                	li	s8,0
1c00bf0e:	000c9463          	bnez	s9,1c00bf16 <_prf+0x496>
1c00bf12:	01a03c33          	snez	s8,s10
1c00bf16:	55f5                	li	a1,-3
1c00bf18:	00bbc663          	blt	s7,a1,1c00bf24 <_prf+0x4a4>
1c00bf1c:	001d0593          	addi	a1,s10,1
1c00bf20:	0b75dd63          	ble	s7,a1,1c00bfda <_prf+0x55a>
1c00bf24:	06700593          	li	a1,103
1c00bf28:	14ba8863          	beq	s5,a1,1c00c078 <_prf+0x5f8>
1c00bf2c:	04500a93          	li	s5,69
1c00bf30:	001d0593          	addi	a1,s10,1
1c00bf34:	4541                	li	a0,16
1c00bf36:	d62a                	sw	a0,44(sp)
1c00bf38:	04a5cdb3          	p.min	s11,a1,a0
1c00bf3c:	a845                	j	1c00bfec <_prf+0x56c>
1c00bf3e:	4c01                	li	s8,0
1c00bf40:	b761                	j	1c00bec8 <_prf+0x448>
1c00bf42:	1808                	addi	a0,sp,48
1c00bf44:	aa3ff0ef          	jal	ra,1c00b9e6 <_rlrshift>
1c00bf48:	5352                	lw	t1,52(sp)
1c00bf4a:	33333737          	lui	a4,0x33333
1c00bf4e:	33270713          	addi	a4,a4,818 # 33333332 <__l2_shared_end+0x1731c192>
1c00bf52:	58c2                	lw	a7,48(sp)
1c00bf54:	0c05                	addi	s8,s8,1
1c00bf56:	fe6766e3          	bltu	a4,t1,1c00bf42 <_prf+0x4c2>
1c00bf5a:	4515                	li	a0,5
1c00bf5c:	031535b3          	mulhu	a1,a0,a7
1c00bf60:	1bfd                	addi	s7,s7,-1
1c00bf62:	031508b3          	mul	a7,a0,a7
1c00bf66:	426505b3          	p.mac	a1,a0,t1
1c00bf6a:	d846                	sw	a7,48(sp)
1c00bf6c:	4501                	li	a0,0
1c00bf6e:	da2e                	sw	a1,52(sp)
1c00bf70:	800007b7          	lui	a5,0x80000
1c00bf74:	fff7c793          	not	a5,a5
1c00bf78:	00b7f663          	bleu	a1,a5,1c00bf84 <_prf+0x504>
1c00bf7c:	d525                	beqz	a0,1c00bee4 <_prf+0x464>
1c00bf7e:	d846                	sw	a7,48(sp)
1c00bf80:	da2e                	sw	a1,52(sp)
1c00bf82:	b78d                	j	1c00bee4 <_prf+0x464>
1c00bf84:	01f8d313          	srli	t1,a7,0x1f
1c00bf88:	00159513          	slli	a0,a1,0x1
1c00bf8c:	00a365b3          	or	a1,t1,a0
1c00bf90:	0886                	slli	a7,a7,0x1
1c00bf92:	1c7d                	addi	s8,s8,-1
1c00bf94:	4505                	li	a0,1
1c00bf96:	bfe9                	j	1c00bf70 <_prf+0x4f0>
1c00bf98:	1808                	addi	a0,sp,48
1c00bf9a:	a6dff0ef          	jal	ra,1c00ba06 <_ldiv5>
1c00bf9e:	58c2                	lw	a7,48(sp)
1c00bfa0:	55d2                	lw	a1,52(sp)
1c00bfa2:	1c7d                	addi	s8,s8,-1
1c00bfa4:	0b85                	addi	s7,s7,1
1c00bfa6:	4501                	li	a0,0
1c00bfa8:	80000737          	lui	a4,0x80000
1c00bfac:	fff74713          	not	a4,a4
1c00bfb0:	00b77663          	bleu	a1,a4,1c00bfbc <_prf+0x53c>
1c00bfb4:	d91d                	beqz	a0,1c00beea <_prf+0x46a>
1c00bfb6:	d846                	sw	a7,48(sp)
1c00bfb8:	da2e                	sw	a1,52(sp)
1c00bfba:	bf05                	j	1c00beea <_prf+0x46a>
1c00bfbc:	01f8d313          	srli	t1,a7,0x1f
1c00bfc0:	00159513          	slli	a0,a1,0x1
1c00bfc4:	00a365b3          	or	a1,t1,a0
1c00bfc8:	0886                	slli	a7,a7,0x1
1c00bfca:	1c7d                	addi	s8,s8,-1
1c00bfcc:	4505                	li	a0,1
1c00bfce:	bfe9                	j	1c00bfa8 <_prf+0x528>
1c00bfd0:	06600593          	li	a1,102
1c00bfd4:	4c01                	li	s8,0
1c00bfd6:	f4ba9de3          	bne	s5,a1,1c00bf30 <_prf+0x4b0>
1c00bfda:	01ab85b3          	add	a1,s7,s10
1c00bfde:	06600a93          	li	s5,102
1c00bfe2:	f405d9e3          	bgez	a1,1c00bf34 <_prf+0x4b4>
1c00bfe6:	45c1                	li	a1,16
1c00bfe8:	d62e                	sw	a1,44(sp)
1c00bfea:	4d81                	li	s11,0
1c00bfec:	4301                	li	t1,0
1c00bfee:	080003b7          	lui	t2,0x8000
1c00bff2:	dc1a                	sw	t1,56(sp)
1c00bff4:	de1e                	sw	t2,60(sp)
1c00bff6:	1dfd                	addi	s11,s11,-1
1c00bff8:	09fdb363          	p.bneimm	s11,-1,1c00c07e <_prf+0x5fe>
1c00bffc:	55c2                	lw	a1,48(sp)
1c00bffe:	5562                	lw	a0,56(sp)
1c00c000:	58d2                	lw	a7,52(sp)
1c00c002:	5372                	lw	t1,60(sp)
1c00c004:	952e                	add	a0,a0,a1
1c00c006:	00b535b3          	sltu	a1,a0,a1
1c00c00a:	989a                	add	a7,a7,t1
1c00c00c:	95c6                	add	a1,a1,a7
1c00c00e:	da2e                	sw	a1,52(sp)
1c00c010:	d82a                	sw	a0,48(sp)
1c00c012:	f605b5b3          	p.bclr	a1,a1,27,0
1c00c016:	c981                	beqz	a1,1c00c026 <_prf+0x5a6>
1c00c018:	1808                	addi	a0,sp,48
1c00c01a:	9edff0ef          	jal	ra,1c00ba06 <_ldiv5>
1c00c01e:	1808                	addi	a0,sp,48
1c00c020:	9c7ff0ef          	jal	ra,1c00b9e6 <_rlrshift>
1c00c024:	0b85                	addi	s7,s7,1
1c00c026:	06600593          	li	a1,102
1c00c02a:	001b0d93          	addi	s11,s6,1
1c00c02e:	08ba9463          	bne	s5,a1,1c00c0b6 <_prf+0x636>
1c00c032:	05705d63          	blez	s7,1c00c08c <_prf+0x60c>
1c00c036:	017b0db3          	add	s11,s6,s7
1c00c03a:	106c                	addi	a1,sp,44
1c00c03c:	1808                	addi	a0,sp,48
1c00c03e:	a0dff0ef          	jal	ra,1c00ba4a <_get_digit>
1c00c042:	00ab00ab          	p.sb	a0,1(s6!)
1c00c046:	ffbb1ae3          	bne	s6,s11,1c00c03a <_prf+0x5ba>
1c00c04a:	4b81                	li	s7,0
1c00c04c:	000c9463          	bnez	s9,1c00c054 <_prf+0x5d4>
1c00c050:	020d0163          	beqz	s10,1c00c072 <_prf+0x5f2>
1c00c054:	001d8b13          	addi	s6,s11,1
1c00c058:	02e00613          	li	a2,46
1c00c05c:	00cd8023          	sb	a2,0(s11)
1c00c060:	8cea                	mv	s9,s10
1c00c062:	8dda                	mv	s11,s6
1c00c064:	03000893          	li	a7,48
1c00c068:	1cfd                	addi	s9,s9,-1
1c00c06a:	03fcb663          	p.bneimm	s9,-1,1c00c096 <_prf+0x616>
1c00c06e:	01ab0db3          	add	s11,s6,s10
1c00c072:	060c1c63          	bnez	s8,1c00c0ea <_prf+0x66a>
1c00c076:	a8c1                	j	1c00c146 <_prf+0x6c6>
1c00c078:	06500a93          	li	s5,101
1c00c07c:	bd55                	j	1c00bf30 <_prf+0x4b0>
1c00c07e:	1828                	addi	a0,sp,56
1c00c080:	987ff0ef          	jal	ra,1c00ba06 <_ldiv5>
1c00c084:	1828                	addi	a0,sp,56
1c00c086:	961ff0ef          	jal	ra,1c00b9e6 <_rlrshift>
1c00c08a:	b7b5                	j	1c00bff6 <_prf+0x576>
1c00c08c:	03000593          	li	a1,48
1c00c090:	00bb0023          	sb	a1,0(s6)
1c00c094:	bf65                	j	1c00c04c <_prf+0x5cc>
1c00c096:	0d85                	addi	s11,s11,1
1c00c098:	000b8663          	beqz	s7,1c00c0a4 <_prf+0x624>
1c00c09c:	ff1d8fa3          	sb	a7,-1(s11)
1c00c0a0:	0b85                	addi	s7,s7,1
1c00c0a2:	b7d9                	j	1c00c068 <_prf+0x5e8>
1c00c0a4:	106c                	addi	a1,sp,44
1c00c0a6:	1808                	addi	a0,sp,48
1c00c0a8:	c446                	sw	a7,8(sp)
1c00c0aa:	9a1ff0ef          	jal	ra,1c00ba4a <_get_digit>
1c00c0ae:	fead8fa3          	sb	a0,-1(s11)
1c00c0b2:	48a2                	lw	a7,8(sp)
1c00c0b4:	bf55                	j	1c00c068 <_prf+0x5e8>
1c00c0b6:	106c                	addi	a1,sp,44
1c00c0b8:	1808                	addi	a0,sp,48
1c00c0ba:	991ff0ef          	jal	ra,1c00ba4a <_get_digit>
1c00c0be:	00ab0023          	sb	a0,0(s6)
1c00c0c2:	03000593          	li	a1,48
1c00c0c6:	00b50363          	beq	a0,a1,1c00c0cc <_prf+0x64c>
1c00c0ca:	1bfd                	addi	s7,s7,-1
1c00c0cc:	000c9463          	bnez	s9,1c00c0d4 <_prf+0x654>
1c00c0d0:	000d0b63          	beqz	s10,1c00c0e6 <_prf+0x666>
1c00c0d4:	002b0d93          	addi	s11,s6,2
1c00c0d8:	02e00593          	li	a1,46
1c00c0dc:	00bb00a3          	sb	a1,1(s6)
1c00c0e0:	9d6e                	add	s10,s10,s11
1c00c0e2:	07bd1863          	bne	s10,s11,1c00c152 <_prf+0x6d2>
1c00c0e6:	000c0f63          	beqz	s8,1c00c104 <_prf+0x684>
1c00c0ea:	03000593          	li	a1,48
1c00c0ee:	fffd8713          	addi	a4,s11,-1
1c00c0f2:	00074603          	lbu	a2,0(a4) # 80000000 <pulp__FC+0x80000001>
1c00c0f6:	06b60563          	beq	a2,a1,1c00c160 <_prf+0x6e0>
1c00c0fa:	02e00593          	li	a1,46
1c00c0fe:	00b61363          	bne	a2,a1,1c00c104 <_prf+0x684>
1c00c102:	8dba                	mv	s11,a4
1c00c104:	c05ab733          	p.bclr	a4,s5,0,5
1c00c108:	04500613          	li	a2,69
1c00c10c:	02c71d63          	bne	a4,a2,1c00c146 <_prf+0x6c6>
1c00c110:	87d6                	mv	a5,s5
1c00c112:	00fd8023          	sb	a5,0(s11)
1c00c116:	02b00793          	li	a5,43
1c00c11a:	000bd663          	bgez	s7,1c00c126 <_prf+0x6a6>
1c00c11e:	41700bb3          	neg	s7,s7
1c00c122:	02d00793          	li	a5,45
1c00c126:	00fd80a3          	sb	a5,1(s11)
1c00c12a:	47a9                	li	a5,10
1c00c12c:	02fbc733          	div	a4,s7,a5
1c00c130:	0d91                	addi	s11,s11,4
1c00c132:	02fbe6b3          	rem	a3,s7,a5
1c00c136:	03070713          	addi	a4,a4,48
1c00c13a:	feed8f23          	sb	a4,-2(s11)
1c00c13e:	03068693          	addi	a3,a3,48
1c00c142:	fedd8fa3          	sb	a3,-1(s11)
1c00c146:	00c8                	addi	a0,sp,68
1c00c148:	000d8023          	sb	zero,0(s11)
1c00c14c:	40ad8533          	sub	a0,s11,a0
1c00c150:	b9e5                	j	1c00be48 <_prf+0x3c8>
1c00c152:	106c                	addi	a1,sp,44
1c00c154:	1808                	addi	a0,sp,48
1c00c156:	8f5ff0ef          	jal	ra,1c00ba4a <_get_digit>
1c00c15a:	00ad80ab          	p.sb	a0,1(s11!)
1c00c15e:	b751                	j	1c00c0e2 <_prf+0x662>
1c00c160:	8dba                	mv	s11,a4
1c00c162:	b771                	j	1c00c0ee <_prf+0x66e>
1c00c164:	000c2783          	lw	a5,0(s8)
1c00c168:	004c0a13          	addi	s4,s8,4
1c00c16c:	0127a023          	sw	s2,0(a5) # 80000000 <pulp__FC+0x80000001>
1c00c170:	b27d                	j	1c00bb1e <_prf+0x9e>
1c00c172:	004c0a13          	addi	s4,s8,4
1c00c176:	000c2583          	lw	a1,0(s8)
1c00c17a:	00dc                	addi	a5,sp,68
1c00c17c:	040c8263          	beqz	s9,1c00c1c0 <_prf+0x740>
1c00c180:	03000693          	li	a3,48
1c00c184:	04d10223          	sb	a3,68(sp)
1c00c188:	04510513          	addi	a0,sp,69
1c00c18c:	e99d                	bnez	a1,1c00c1c2 <_prf+0x742>
1c00c18e:	040102a3          	sb	zero,69(sp)
1c00c192:	4401                	li	s0,0
1c00c194:	0dfd3063          	p.bneimm	s10,-1,1c00c254 <_prf+0x7d4>
1c00c198:	04410b93          	addi	s7,sp,68
1c00c19c:	0d3cc063          	blt	s9,s3,1c00c25c <_prf+0x7dc>
1c00c1a0:	89e6                	mv	s3,s9
1c00c1a2:	41790433          	sub	s0,s2,s7
1c00c1a6:	01740933          	add	s2,s0,s7
1c00c1aa:	96098ae3          	beqz	s3,1c00bb1e <_prf+0x9e>
1c00c1ae:	45f2                	lw	a1,28(sp)
1c00c1b0:	001bc50b          	p.lbu	a0,1(s7!)
1c00c1b4:	47e2                	lw	a5,24(sp)
1c00c1b6:	9782                	jalr	a5
1c00c1b8:	93f522e3          	p.beqimm	a0,-1,1c00badc <_prf+0x5c>
1c00c1bc:	19fd                	addi	s3,s3,-1
1c00c1be:	b7e5                	j	1c00c1a6 <_prf+0x726>
1c00c1c0:	853e                	mv	a0,a5
1c00c1c2:	86ea                	mv	a3,s10
1c00c1c4:	4621                	li	a2,8
1c00c1c6:	40f50433          	sub	s0,a0,a5
1c00c1ca:	fc6ff0ef          	jal	ra,1c00b990 <_to_x>
1c00c1ce:	9522                	add	a0,a0,s0
1c00c1d0:	4401                	li	s0,0
1c00c1d2:	c9fd25e3          	p.beqimm	s10,-1,1c00be5c <_prf+0x3dc>
1c00c1d6:	02000793          	li	a5,32
1c00c1da:	c63e                	sw	a5,12(sp)
1c00c1dc:	b141                	j	1c00be5c <_prf+0x3dc>
1c00c1de:	000c2583          	lw	a1,0(s8)
1c00c1e2:	77e1                	lui	a5,0xffff8
1c00c1e4:	8307c793          	xori	a5,a5,-2000
1c00c1e8:	46a1                	li	a3,8
1c00c1ea:	4641                	li	a2,16
1c00c1ec:	04610513          	addi	a0,sp,70
1c00c1f0:	04f11223          	sh	a5,68(sp)
1c00c1f4:	f9cff0ef          	jal	ra,1c00b990 <_to_x>
1c00c1f8:	004c0a13          	addi	s4,s8,4
1c00c1fc:	0509                	addi	a0,a0,2
1c00c1fe:	4401                	li	s0,0
1c00c200:	be5d                	j	1c00bdb6 <_prf+0x336>
1c00c202:	000d4463          	bltz	s10,1c00c20a <_prf+0x78a>
1c00c206:	05acccb3          	p.min	s9,s9,s10
1c00c20a:	900c8ae3          	beqz	s9,1c00bb1e <_prf+0x9e>
1c00c20e:	8666                	mv	a2,s9
1c00c210:	00c8                	addi	a0,sp,68
1c00c212:	d0aff0ef          	jal	ra,1c00b71c <memcpy>
1c00c216:	b6ad                	j	1c00bd80 <_prf+0x300>
1c00c218:	000c2583          	lw	a1,0(s8)
1c00c21c:	86ea                	mv	a3,s10
1c00c21e:	4629                	li	a2,10
1c00c220:	00c8                	addi	a0,sp,68
1c00c222:	004c0a13          	addi	s4,s8,4
1c00c226:	f6aff0ef          	jal	ra,1c00b990 <_to_x>
1c00c22a:	b75d                	j	1c00c1d0 <_prf+0x750>
1c00c22c:	f9f78613          	addi	a2,a5,-97 # ffff7f9f <pulp__FC+0xffff7fa0>
1c00c230:	0ff67613          	andi	a2,a2,255
1c00c234:	aac5e1e3          	bltu	a1,a2,1c00bcd6 <_prf+0x256>
1c00c238:	1781                	addi	a5,a5,-32
1c00c23a:	fef68fa3          	sb	a5,-1(a3)
1c00c23e:	bc61                	j	1c00bcd6 <_prf+0x256>
1c00c240:	45f2                	lw	a1,28(sp)
1c00c242:	4762                	lw	a4,24(sp)
1c00c244:	02500513          	li	a0,37
1c00c248:	9702                	jalr	a4
1c00c24a:	89f529e3          	p.beqimm	a0,-1,1c00badc <_prf+0x5c>
1c00c24e:	0905                	addi	s2,s2,1
1c00c250:	8a62                	mv	s4,s8
1c00c252:	b0f1                	j	1c00bb1e <_prf+0x9e>
1c00c254:	02000793          	li	a5,32
1c00c258:	c63e                	sw	a5,12(sp)
1c00c25a:	bf3d                	j	1c00c198 <_prf+0x718>
1c00c25c:	4752                	lw	a4,20(sp)
1c00c25e:	cf01                	beqz	a4,1c00c276 <_prf+0x7f6>
1c00c260:	019b8833          	add	a6,s7,s9
1c00c264:	02000713          	li	a4,32
1c00c268:	417807b3          	sub	a5,a6,s7
1c00c26c:	f337dbe3          	ble	s3,a5,1c00c1a2 <_prf+0x722>
1c00c270:	00e800ab          	p.sb	a4,1(a6!)
1c00c274:	bfd5                	j	1c00c268 <_prf+0x7e8>
1c00c276:	41998c33          	sub	s8,s3,s9
1c00c27a:	001c8613          	addi	a2,s9,1
1c00c27e:	85de                	mv	a1,s7
1c00c280:	018b8533          	add	a0,s7,s8
1c00c284:	cceff0ef          	jal	ra,1c00b752 <memmove>
1c00c288:	4732                	lw	a4,12(sp)
1c00c28a:	02000793          	li	a5,32
1c00c28e:	00f70363          	beq	a4,a5,1c00c294 <_prf+0x814>
1c00c292:	ca22                	sw	s0,20(sp)
1c00c294:	47d2                	lw	a5,20(sp)
1c00c296:	9c3e                	add	s8,s8,a5
1c00c298:	00fb8ab3          	add	s5,s7,a5
1c00c29c:	417a87b3          	sub	a5,s5,s7
1c00c2a0:	f187d1e3          	ble	s8,a5,1c00c1a2 <_prf+0x722>
1c00c2a4:	4732                	lw	a4,12(sp)
1c00c2a6:	00ea80ab          	p.sb	a4,1(s5!)
1c00c2aa:	bfcd                	j	1c00c29c <_prf+0x81c>
1c00c2ac:	4442                	lw	s0,16(sp)
1c00c2ae:	b67d                	j	1c00be5c <_prf+0x3dc>

1c00c2b0 <__rt_uart_cluster_req_done>:
1c00c2b0:	300476f3          	csrrci	a3,mstatus,8
1c00c2b4:	4785                	li	a5,1
1c00c2b6:	08f50c23          	sb	a5,152(a0)
1c00c2ba:	09954783          	lbu	a5,153(a0)
1c00c2be:	00201737          	lui	a4,0x201
1c00c2c2:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e7e1c>
1c00c2c6:	04078793          	addi	a5,a5,64
1c00c2ca:	07da                	slli	a5,a5,0x16
1c00c2cc:	0007e723          	p.sw	zero,a4(a5)
1c00c2d0:	30069073          	csrw	mstatus,a3
1c00c2d4:	8082                	ret

1c00c2d6 <__rt_uart_cluster_req>:
1c00c2d6:	1141                	addi	sp,sp,-16
1c00c2d8:	c606                	sw	ra,12(sp)
1c00c2da:	c422                	sw	s0,8(sp)
1c00c2dc:	30047473          	csrrci	s0,mstatus,8
1c00c2e0:	1c00c7b7          	lui	a5,0x1c00c
1c00c2e4:	2b078793          	addi	a5,a5,688 # 1c00c2b0 <__rt_uart_cluster_req_done>
1c00c2e8:	c55c                	sw	a5,12(a0)
1c00c2ea:	4785                	li	a5,1
1c00c2ec:	d55c                	sw	a5,44(a0)
1c00c2ee:	411c                	lw	a5,0(a0)
1c00c2f0:	02052823          	sw	zero,48(a0)
1c00c2f4:	c908                	sw	a0,16(a0)
1c00c2f6:	43cc                	lw	a1,4(a5)
1c00c2f8:	4514                	lw	a3,8(a0)
1c00c2fa:	4150                	lw	a2,4(a0)
1c00c2fc:	0586                	slli	a1,a1,0x1
1c00c2fe:	00c50793          	addi	a5,a0,12
1c00c302:	4701                	li	a4,0
1c00c304:	0585                	addi	a1,a1,1
1c00c306:	4501                	li	a0,0
1c00c308:	f07fe0ef          	jal	ra,1c00b20e <rt_periph_copy>
1c00c30c:	30041073          	csrw	mstatus,s0
1c00c310:	40b2                	lw	ra,12(sp)
1c00c312:	4422                	lw	s0,8(sp)
1c00c314:	0141                	addi	sp,sp,16
1c00c316:	8082                	ret

1c00c318 <__rt_uart_wait_tx_done.isra.2>:
1c00c318:	1a1026b7          	lui	a3,0x1a102
1c00c31c:	09068693          	addi	a3,a3,144 # 1a102090 <__l1_end+0xa0fb078>
1c00c320:	411c                	lw	a5,0(a0)
1c00c322:	079e                	slli	a5,a5,0x7
1c00c324:	00d78733          	add	a4,a5,a3
1c00c328:	0721                	addi	a4,a4,8
1c00c32a:	4318                	lw	a4,0(a4)
1c00c32c:	8b41                	andi	a4,a4,16
1c00c32e:	ef39                	bnez	a4,1c00c38c <__rt_uart_wait_tx_done.isra.2+0x74>
1c00c330:	1a102737          	lui	a4,0x1a102
1c00c334:	0a070713          	addi	a4,a4,160 # 1a1020a0 <__l1_end+0xa0fb088>
1c00c338:	97ba                	add	a5,a5,a4
1c00c33a:	4398                	lw	a4,0(a5)
1c00c33c:	fc173733          	p.bclr	a4,a4,30,1
1c00c340:	ff6d                	bnez	a4,1c00c33a <__rt_uart_wait_tx_done.isra.2+0x22>
1c00c342:	f14027f3          	csrr	a5,mhartid
1c00c346:	8795                	srai	a5,a5,0x5
1c00c348:	1a109737          	lui	a4,0x1a109
1c00c34c:	00204637          	lui	a2,0x204
1c00c350:	f267b7b3          	p.bclr	a5,a5,25,6
1c00c354:	01470813          	addi	a6,a4,20 # 1a109014 <__l1_end+0xa101ffc>
1c00c358:	01460e13          	addi	t3,a2,20 # 204014 <__l1_heap_size+0x1eb02c>
1c00c35c:	00470e93          	addi	t4,a4,4
1c00c360:	6691                	lui	a3,0x4
1c00c362:	6311                	lui	t1,0x4
1c00c364:	457d                	li	a0,31
1c00c366:	88be                	mv	a7,a5
1c00c368:	0641                	addi	a2,a2,16
1c00c36a:	0721                	addi	a4,a4,8
1c00c36c:	03200593          	li	a1,50
1c00c370:	00682023          	sw	t1,0(a6)
1c00c374:	00a79f63          	bne	a5,a0,1c00c392 <__rt_uart_wait_tx_done.isra.2+0x7a>
1c00c378:	00dea023          	sw	a3,0(t4)
1c00c37c:	10500073          	wfi
1c00c380:	00a89c63          	bne	a7,a0,1c00c398 <__rt_uart_wait_tx_done.isra.2+0x80>
1c00c384:	c314                	sw	a3,0(a4)
1c00c386:	15fd                	addi	a1,a1,-1
1c00c388:	f5e5                	bnez	a1,1c00c370 <__rt_uart_wait_tx_done.isra.2+0x58>
1c00c38a:	8082                	ret
1c00c38c:	10500073          	wfi
1c00c390:	bf41                	j	1c00c320 <__rt_uart_wait_tx_done.isra.2+0x8>
1c00c392:	00de2023          	sw	a3,0(t3)
1c00c396:	b7dd                	j	1c00c37c <__rt_uart_wait_tx_done.isra.2+0x64>
1c00c398:	c214                	sw	a3,0(a2)
1c00c39a:	b7f5                	j	1c00c386 <__rt_uart_wait_tx_done.isra.2+0x6e>

1c00c39c <__rt_uart_setup>:
1c00c39c:	4518                	lw	a4,8(a0)
1c00c39e:	1c0016b7          	lui	a3,0x1c001
1c00c3a2:	7e86a683          	lw	a3,2024(a3) # 1c0017e8 <__rt_freq_domains>
1c00c3a6:	00175793          	srli	a5,a4,0x1
1c00c3aa:	97b6                	add	a5,a5,a3
1c00c3ac:	02e7d7b3          	divu	a5,a5,a4
1c00c3b0:	4154                	lw	a3,4(a0)
1c00c3b2:	1a102737          	lui	a4,0x1a102
1c00c3b6:	0a470713          	addi	a4,a4,164 # 1a1020a4 <__l1_end+0xa0fb08c>
1c00c3ba:	069e                	slli	a3,a3,0x7
1c00c3bc:	17fd                	addi	a5,a5,-1
1c00c3be:	07c2                	slli	a5,a5,0x10
1c00c3c0:	3067e793          	ori	a5,a5,774
1c00c3c4:	00f6e723          	p.sw	a5,a4(a3)
1c00c3c8:	8082                	ret

1c00c3ca <__rt_uart_setfreq_after>:
1c00c3ca:	1c001537          	lui	a0,0x1c001
1c00c3ce:	70852783          	lw	a5,1800(a0) # 1c001708 <__rt_uart>
1c00c3d2:	1141                	addi	sp,sp,-16
1c00c3d4:	c422                	sw	s0,8(sp)
1c00c3d6:	c606                	sw	ra,12(sp)
1c00c3d8:	70850413          	addi	s0,a0,1800
1c00c3dc:	c781                	beqz	a5,1c00c3e4 <__rt_uart_setfreq_after+0x1a>
1c00c3de:	70850513          	addi	a0,a0,1800
1c00c3e2:	3f6d                	jal	1c00c39c <__rt_uart_setup>
1c00c3e4:	481c                	lw	a5,16(s0)
1c00c3e6:	c781                	beqz	a5,1c00c3ee <__rt_uart_setfreq_after+0x24>
1c00c3e8:	01040513          	addi	a0,s0,16
1c00c3ec:	3f45                	jal	1c00c39c <__rt_uart_setup>
1c00c3ee:	501c                	lw	a5,32(s0)
1c00c3f0:	c781                	beqz	a5,1c00c3f8 <__rt_uart_setfreq_after+0x2e>
1c00c3f2:	02040513          	addi	a0,s0,32
1c00c3f6:	375d                	jal	1c00c39c <__rt_uart_setup>
1c00c3f8:	40b2                	lw	ra,12(sp)
1c00c3fa:	4422                	lw	s0,8(sp)
1c00c3fc:	4501                	li	a0,0
1c00c3fe:	0141                	addi	sp,sp,16
1c00c400:	8082                	ret

1c00c402 <soc_eu_fcEventMask_setEvent>:
1c00c402:	02000793          	li	a5,32
1c00c406:	02f54733          	div	a4,a0,a5
1c00c40a:	1a1066b7          	lui	a3,0x1a106
1c00c40e:	0691                	addi	a3,a3,4
1c00c410:	02f56533          	rem	a0,a0,a5
1c00c414:	070a                	slli	a4,a4,0x2
1c00c416:	9736                	add	a4,a4,a3
1c00c418:	4314                	lw	a3,0(a4)
1c00c41a:	4785                	li	a5,1
1c00c41c:	00a797b3          	sll	a5,a5,a0
1c00c420:	fff7c793          	not	a5,a5
1c00c424:	8ff5                	and	a5,a5,a3
1c00c426:	c31c                	sw	a5,0(a4)
1c00c428:	8082                	ret

1c00c42a <__rt_uart_setfreq_before>:
1c00c42a:	1101                	addi	sp,sp,-32
1c00c42c:	cc22                	sw	s0,24(sp)
1c00c42e:	c84a                	sw	s2,16(sp)
1c00c430:	c64e                	sw	s3,12(sp)
1c00c432:	1c001437          	lui	s0,0x1c001
1c00c436:	1a102937          	lui	s2,0x1a102
1c00c43a:	005009b7          	lui	s3,0x500
1c00c43e:	ca26                	sw	s1,20(sp)
1c00c440:	ce06                	sw	ra,28(sp)
1c00c442:	70840413          	addi	s0,s0,1800 # 1c001708 <__rt_uart>
1c00c446:	4481                	li	s1,0
1c00c448:	0a490913          	addi	s2,s2,164 # 1a1020a4 <__l1_end+0xa0fb08c>
1c00c44c:	0999                	addi	s3,s3,6
1c00c44e:	401c                	lw	a5,0(s0)
1c00c450:	cb81                	beqz	a5,1c00c460 <__rt_uart_setfreq_before+0x36>
1c00c452:	00440513          	addi	a0,s0,4
1c00c456:	35c9                	jal	1c00c318 <__rt_uart_wait_tx_done.isra.2>
1c00c458:	405c                	lw	a5,4(s0)
1c00c45a:	079e                	slli	a5,a5,0x7
1c00c45c:	0137e923          	p.sw	s3,s2(a5)
1c00c460:	0485                	addi	s1,s1,1
1c00c462:	0441                	addi	s0,s0,16
1c00c464:	fe34b5e3          	p.bneimm	s1,3,1c00c44e <__rt_uart_setfreq_before+0x24>
1c00c468:	40f2                	lw	ra,28(sp)
1c00c46a:	4462                	lw	s0,24(sp)
1c00c46c:	44d2                	lw	s1,20(sp)
1c00c46e:	4942                	lw	s2,16(sp)
1c00c470:	49b2                	lw	s3,12(sp)
1c00c472:	4501                	li	a0,0
1c00c474:	6105                	addi	sp,sp,32
1c00c476:	8082                	ret

1c00c478 <rt_uart_conf_init>:
1c00c478:	000997b7          	lui	a5,0x99
1c00c47c:	96878793          	addi	a5,a5,-1688 # 98968 <__l1_heap_size+0x7f980>
1c00c480:	c11c                	sw	a5,0(a0)
1c00c482:	57fd                	li	a5,-1
1c00c484:	c15c                	sw	a5,4(a0)
1c00c486:	8082                	ret

1c00c488 <__rt_uart_open>:
1c00c488:	1141                	addi	sp,sp,-16
1c00c48a:	c606                	sw	ra,12(sp)
1c00c48c:	c422                	sw	s0,8(sp)
1c00c48e:	c226                	sw	s1,4(sp)
1c00c490:	c04a                	sw	s2,0(sp)
1c00c492:	30047973          	csrrci	s2,mstatus,8
1c00c496:	cd8d                	beqz	a1,1c00c4d0 <__rt_uart_open+0x48>
1c00c498:	4198                	lw	a4,0(a1)
1c00c49a:	1c0016b7          	lui	a3,0x1c001
1c00c49e:	ffc50793          	addi	a5,a0,-4
1c00c4a2:	70868413          	addi	s0,a3,1800 # 1c001708 <__rt_uart>
1c00c4a6:	0792                	slli	a5,a5,0x4
1c00c4a8:	943e                	add	s0,s0,a5
1c00c4aa:	4010                	lw	a2,0(s0)
1c00c4ac:	70868693          	addi	a3,a3,1800
1c00c4b0:	c60d                	beqz	a2,1c00c4da <__rt_uart_open+0x52>
1c00c4b2:	c589                	beqz	a1,1c00c4bc <__rt_uart_open+0x34>
1c00c4b4:	418c                	lw	a1,0(a1)
1c00c4b6:	4418                	lw	a4,8(s0)
1c00c4b8:	04e59863          	bne	a1,a4,1c00c508 <__rt_uart_open+0x80>
1c00c4bc:	0605                	addi	a2,a2,1
1c00c4be:	00c6e7a3          	p.sw	a2,a5(a3)
1c00c4c2:	8522                	mv	a0,s0
1c00c4c4:	40b2                	lw	ra,12(sp)
1c00c4c6:	4422                	lw	s0,8(sp)
1c00c4c8:	4492                	lw	s1,4(sp)
1c00c4ca:	4902                	lw	s2,0(sp)
1c00c4cc:	0141                	addi	sp,sp,16
1c00c4ce:	8082                	ret
1c00c4d0:	00099737          	lui	a4,0x99
1c00c4d4:	96870713          	addi	a4,a4,-1688 # 98968 <__l1_heap_size+0x7f980>
1c00c4d8:	b7c9                	j	1c00c49a <__rt_uart_open+0x12>
1c00c4da:	4785                	li	a5,1
1c00c4dc:	c01c                	sw	a5,0(s0)
1c00c4de:	c418                	sw	a4,8(s0)
1c00c4e0:	c048                	sw	a0,4(s0)
1c00c4e2:	1a102737          	lui	a4,0x1a102
1c00c4e6:	4314                	lw	a3,0(a4)
1c00c4e8:	00a797b3          	sll	a5,a5,a0
1c00c4ec:	00251493          	slli	s1,a0,0x2
1c00c4f0:	8fd5                	or	a5,a5,a3
1c00c4f2:	c31c                	sw	a5,0(a4)
1c00c4f4:	8526                	mv	a0,s1
1c00c4f6:	3731                	jal	1c00c402 <soc_eu_fcEventMask_setEvent>
1c00c4f8:	00148513          	addi	a0,s1,1
1c00c4fc:	3719                	jal	1c00c402 <soc_eu_fcEventMask_setEvent>
1c00c4fe:	8522                	mv	a0,s0
1c00c500:	3d71                	jal	1c00c39c <__rt_uart_setup>
1c00c502:	30091073          	csrw	mstatus,s2
1c00c506:	bf75                	j	1c00c4c2 <__rt_uart_open+0x3a>
1c00c508:	4401                	li	s0,0
1c00c50a:	bf65                	j	1c00c4c2 <__rt_uart_open+0x3a>

1c00c50c <rt_uart_close>:
1c00c50c:	1141                	addi	sp,sp,-16
1c00c50e:	c606                	sw	ra,12(sp)
1c00c510:	c422                	sw	s0,8(sp)
1c00c512:	c226                	sw	s1,4(sp)
1c00c514:	300474f3          	csrrci	s1,mstatus,8
1c00c518:	411c                	lw	a5,0(a0)
1c00c51a:	17fd                	addi	a5,a5,-1
1c00c51c:	c11c                	sw	a5,0(a0)
1c00c51e:	eb85                	bnez	a5,1c00c54e <rt_uart_close+0x42>
1c00c520:	842a                	mv	s0,a0
1c00c522:	0511                	addi	a0,a0,4
1c00c524:	3bd5                	jal	1c00c318 <__rt_uart_wait_tx_done.isra.2>
1c00c526:	405c                	lw	a5,4(s0)
1c00c528:	1a102737          	lui	a4,0x1a102
1c00c52c:	00500637          	lui	a2,0x500
1c00c530:	079e                	slli	a5,a5,0x7
1c00c532:	0a470693          	addi	a3,a4,164 # 1a1020a4 <__l1_end+0xa0fb08c>
1c00c536:	0619                	addi	a2,a2,6
1c00c538:	00c7e6a3          	p.sw	a2,a3(a5)
1c00c53c:	4050                	lw	a2,4(s0)
1c00c53e:	4314                	lw	a3,0(a4)
1c00c540:	4785                	li	a5,1
1c00c542:	00c797b3          	sll	a5,a5,a2
1c00c546:	fff7c793          	not	a5,a5
1c00c54a:	8ff5                	and	a5,a5,a3
1c00c54c:	c31c                	sw	a5,0(a4)
1c00c54e:	30049073          	csrw	mstatus,s1
1c00c552:	40b2                	lw	ra,12(sp)
1c00c554:	4422                	lw	s0,8(sp)
1c00c556:	4492                	lw	s1,4(sp)
1c00c558:	0141                	addi	sp,sp,16
1c00c55a:	8082                	ret

1c00c55c <rt_uart_cluster_write>:
1c00c55c:	f14027f3          	csrr	a5,mhartid
1c00c560:	8795                	srai	a5,a5,0x5
1c00c562:	f267b7b3          	p.bclr	a5,a5,25,6
1c00c566:	08f68ca3          	sb	a5,153(a3)
1c00c56a:	1c00c7b7          	lui	a5,0x1c00c
1c00c56e:	2d678793          	addi	a5,a5,726 # 1c00c2d6 <__rt_uart_cluster_req>
1c00c572:	c6dc                	sw	a5,12(a3)
1c00c574:	4785                	li	a5,1
1c00c576:	c288                	sw	a0,0(a3)
1c00c578:	c2cc                	sw	a1,4(a3)
1c00c57a:	c690                	sw	a2,8(a3)
1c00c57c:	08068c23          	sb	zero,152(a3)
1c00c580:	0206a823          	sw	zero,48(a3)
1c00c584:	ca94                	sw	a3,16(a3)
1c00c586:	d6dc                	sw	a5,44(a3)
1c00c588:	00c68513          	addi	a0,a3,12
1c00c58c:	dadfd06f          	j	1c00a338 <__rt_cluster_push_fc_event>

1c00c590 <__rt_uart_init>:
1c00c590:	1c00c5b7          	lui	a1,0x1c00c
1c00c594:	1141                	addi	sp,sp,-16
1c00c596:	4601                	li	a2,0
1c00c598:	42a58593          	addi	a1,a1,1066 # 1c00c42a <__rt_uart_setfreq_before>
1c00c59c:	4511                	li	a0,4
1c00c59e:	c606                	sw	ra,12(sp)
1c00c5a0:	c422                	sw	s0,8(sp)
1c00c5a2:	dc8fe0ef          	jal	ra,1c00ab6a <__rt_cbsys_add>
1c00c5a6:	1c00c5b7          	lui	a1,0x1c00c
1c00c5aa:	842a                	mv	s0,a0
1c00c5ac:	4601                	li	a2,0
1c00c5ae:	3ca58593          	addi	a1,a1,970 # 1c00c3ca <__rt_uart_setfreq_after>
1c00c5b2:	4515                	li	a0,5
1c00c5b4:	db6fe0ef          	jal	ra,1c00ab6a <__rt_cbsys_add>
1c00c5b8:	1c0017b7          	lui	a5,0x1c001
1c00c5bc:	70878793          	addi	a5,a5,1800 # 1c001708 <__rt_uart>
1c00c5c0:	0007a023          	sw	zero,0(a5)
1c00c5c4:	0007a823          	sw	zero,16(a5)
1c00c5c8:	0207a023          	sw	zero,32(a5)
1c00c5cc:	8d41                	or	a0,a0,s0
1c00c5ce:	c10d                	beqz	a0,1c00c5f0 <__rt_uart_init+0x60>
1c00c5d0:	f1402673          	csrr	a2,mhartid
1c00c5d4:	1c001537          	lui	a0,0x1c001
1c00c5d8:	40565593          	srai	a1,a2,0x5
1c00c5dc:	f265b5b3          	p.bclr	a1,a1,25,6
1c00c5e0:	f4563633          	p.bclr	a2,a2,26,5
1c00c5e4:	b9450513          	addi	a0,a0,-1132 # 1c000b94 <PIo2+0x238>
1c00c5e8:	b7eff0ef          	jal	ra,1c00b966 <printf>
1c00c5ec:	b08ff0ef          	jal	ra,1c00b8f4 <abort>
1c00c5f0:	40b2                	lw	ra,12(sp)
1c00c5f2:	4422                	lw	s0,8(sp)
1c00c5f4:	0141                	addi	sp,sp,16
1c00c5f6:	8082                	ret

Disassembly of section .l2_data:

1c010000 <__cluster_text_start>:
1c010000:	f1402573          	csrr	a0,mhartid
1c010004:	01f57593          	andi	a1,a0,31
1c010008:	8115                	srli	a0,a0,0x5
1c01000a:	000702b7          	lui	t0,0x70
1c01000e:	00204337          	lui	t1,0x204
1c010012:	00532023          	sw	t0,0(t1) # 204000 <__l1_heap_size+0x1eb018>
1c010016:	43a1                	li	t2,8
1c010018:	10759063          	bne	a1,t2,1c010118 <__rt_slave_start>
1c01001c:	e3ff0417          	auipc	s0,0xe3ff0
1c010020:	fec40413          	addi	s0,s0,-20 # 8 <__rt_first_free>
1c010024:	002049b7          	lui	s3,0x204
1c010028:	4a09                	li	s4,2
1c01002a:	00000a97          	auipc	s5,0x0
1c01002e:	034a8a93          	addi	s5,s5,52 # 1c01005e <__rt_master_event>
1c010032:	ffff1b97          	auipc	s7,0xffff1
1c010036:	782b8b93          	addi	s7,s7,1922 # 1c0017b4 <__rt_fc_cluster_data>
1c01003a:	02800393          	li	t2,40
1c01003e:	02a383b3          	mul	t2,t2,a0
1c010042:	9b9e                	add	s7,s7,t2
1c010044:	0b91                	addi	s7,s7,4
1c010046:	1a109cb7          	lui	s9,0x1a109
1c01004a:	010c8c93          	addi	s9,s9,16 # 1a109010 <__l1_end+0xa101ff8>
1c01004e:	4c09                	li	s8,2
1c010050:	00000d17          	auipc	s10,0x0
1c010054:	106d0d13          	addi	s10,s10,262 # 1c010156 <__rt_set_slave_stack>
1c010058:	001d6d13          	ori	s10,s10,1
1c01005c:	a819                	j	1c010072 <__rt_master_loop>

1c01005e <__rt_master_event>:
1c01005e:	000b0a63          	beqz	s6,1c010072 <__rt_master_loop>

1c010062 <__rt_push_event_to_fc_retry>:
1c010062:	000ba283          	lw	t0,0(s7)
1c010066:	0a029263          	bnez	t0,1c01010a <__rt_push_event_to_fc_wait>
1c01006a:	016ba023          	sw	s6,0(s7)
1c01006e:	018ca023          	sw	s8,0(s9)

1c010072 <__rt_master_loop>:
1c010072:	00042e03          	lw	t3,0(s0)
1c010076:	080e0363          	beqz	t3,1c0100fc <__rt_master_sleep>

1c01007a <__rt_master_loop_update_next>:
1c01007a:	020e2e83          	lw	t4,32(t3)
1c01007e:	020e2223          	sw	zero,36(t3)
1c010082:	01d42023          	sw	t4,0(s0)
1c010086:	020e2f03          	lw	t5,32(t3)
1c01008a:	ffee98e3          	bne	t4,t5,1c01007a <__rt_master_loop_update_next>
1c01008e:	7d005073          	csrwi	0x7d0,0
1c010092:	004e2503          	lw	a0,4(t3)
1c010096:	000e2283          	lw	t0,0(t3)
1c01009a:	008e2103          	lw	sp,8(t3)
1c01009e:	00ce2303          	lw	t1,12(t3)
1c0100a2:	010e2383          	lw	t2,16(t3)
1c0100a6:	028e2f03          	lw	t5,40(t3)
1c0100aa:	018e2b03          	lw	s6,24(t3)
1c0100ae:	014e2f83          	lw	t6,20(t3)
1c0100b2:	80d6                	mv	ra,s5
1c0100b4:	911a                	add	sp,sp,t1
1c0100b6:	01f02a23          	sw	t6,20(zero) # 14 <__rt_cluster_nb_active_pe>
1c0100ba:	00030a63          	beqz	t1,1c0100ce <__rt_no_stack_check>
1c0100be:	40610eb3          	sub	t4,sp,t1
1c0100c2:	7d1e9073          	csrw	0x7d1,t4
1c0100c6:	7d211073          	csrw	0x7d2,sp
1c0100ca:	7d00d073          	csrwi	0x7d0,1

1c0100ce <__rt_no_stack_check>:
1c0100ce:	09e9a223          	sw	t5,132(s3) # 204084 <__l1_heap_size+0x1eb09c>
1c0100d2:	000f0663          	beqz	t5,1c0100de <__rt_master_no_slave_barrier>
1c0100d6:	21e9a023          	sw	t5,512(s3)
1c0100da:	21e9a623          	sw	t5,524(s3)

1c0100de <__rt_master_no_slave_barrier>:
1c0100de:	100f6f93          	ori	t6,t5,256
1c0100e2:	23f9a023          	sw	t6,544(s3)
1c0100e6:	23f9a623          	sw	t6,556(s3)
1c0100ea:	000f2863          	p.beqimm	t5,0,1c0100fa <__rt_master_loop_no_slave>
1c0100ee:	09a9a023          	sw	s10,128(s3)
1c0100f2:	0879a023          	sw	t2,128(s3)
1c0100f6:	0829a023          	sw	sp,128(s3)

1c0100fa <__rt_master_loop_no_slave>:
1c0100fa:	8282                	jr	t0

1c0100fc <__rt_master_sleep>:
1c0100fc:	0149a423          	sw	s4,8(s3)
1c010100:	03c9e003          	p.elw	zero,60(s3)
1c010104:	0149a223          	sw	s4,4(s3)
1c010108:	b7ad                	j	1c010072 <__rt_master_loop>

1c01010a <__rt_push_event_to_fc_wait>:
1c01010a:	0149a423          	sw	s4,8(s3)
1c01010e:	03c9e003          	p.elw	zero,60(s3)
1c010112:	0149a223          	sw	s4,4(s3)
1c010116:	b7b1                	j	1c010062 <__rt_push_event_to_fc_retry>

1c010118 <__rt_slave_start>:
1c010118:	00204937          	lui	s2,0x204
1c01011c:	f14029f3          	csrr	s3,mhartid
1c010120:	01f9f993          	andi	s3,s3,31
1c010124:	00000a17          	auipc	s4,0x0
1c010128:	012a0a13          	addi	s4,s4,18 # 1c010136 <__rt_fork_return>
1c01012c:	00000a97          	auipc	s5,0x0
1c010130:	00ea8a93          	addi	s5,s5,14 # 1c01013a <__rt_wait_for_dispatch>
1c010134:	a019                	j	1c01013a <__rt_wait_for_dispatch>

1c010136 <__rt_fork_return>:
1c010136:	23c96283          	p.elw	t0,572(s2) # 20423c <__l1_heap_size+0x1eb254>

1c01013a <__rt_wait_for_dispatch>:
1c01013a:	08096283          	p.elw	t0,128(s2)
1c01013e:	08096503          	p.elw	a0,128(s2)
1c010142:	0012f313          	andi	t1,t0,1
1c010146:	00031563          	bnez	t1,1c010150 <__rt_other_entry>

1c01014a <__rt_fork_entry>:
1c01014a:	000a00b3          	add	ra,s4,zero
1c01014e:	8282                	jr	t0

1c010150 <__rt_other_entry>:
1c010150:	000a80b3          	add	ra,s5,zero
1c010154:	8282                	jr	t0

1c010156 <__rt_set_slave_stack>:
1c010156:	7d005073          	csrwi	0x7d0,0
1c01015a:	08096283          	p.elw	t0,128(s2)
1c01015e:	00198f13          	addi	t5,s3,1
1c010162:	02af0eb3          	mul	t4,t5,a0
1c010166:	005e8133          	add	sp,t4,t0
1c01016a:	c909                	beqz	a0,1c01017c <__rt_no_stack_check_end>
1c01016c:	40a10eb3          	sub	t4,sp,a0
1c010170:	7d1e9073          	csrw	0x7d1,t4
1c010174:	7d211073          	csrw	0x7d2,sp
1c010178:	7d00d073          	csrwi	0x7d0,1

1c01017c <__rt_no_stack_check_end>:
1c01017c:	8082                	ret
	...

1c010180 <__cluster_text_end>:
	...
