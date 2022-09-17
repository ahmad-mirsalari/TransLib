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
1c0080ac:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x2347ce38>
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
1c008108:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x2347ce38>
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
1c00816e:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x2347ce38>
1c008172:	00874763          	blt	a4,s0,1c008180 <__ieee754_rem_pio2f+0x24>
1c008176:	c188                	sw	a0,0(a1)
1c008178:	0005a223          	sw	zero,4(a1)
1c00817c:	4501                	li	a0,0
1c00817e:	a0a9                	j	1c0081c8 <__ieee754_rem_pio2f+0x6c>
1c008180:	4016d737          	lui	a4,0x4016d
1c008184:	be370713          	addi	a4,a4,-1053 # 4016cbe3 <__l2_shared_end+0x24158a43>
1c008188:	892a                	mv	s2,a0
1c00818a:	0a874163          	blt	a4,s0,1c00822c <__ieee754_rem_pio2f+0xd0>
1c00818e:	1c001737          	lui	a4,0x1c001
1c008192:	c6043433          	p.bclr	s0,s0,3,0
1c008196:	be872703          	lw	a4,-1048(a4) # 1c000be8 <PIo2+0x28c>
1c00819a:	04a05863          	blez	a0,1c0081ea <__ieee754_rem_pio2f+0x8e>
1c00819e:	08e577d3          	fsub.s	a5,a0,a4
1c0081a2:	3fc91737          	lui	a4,0x3fc91
1c0081a6:	fd070713          	addi	a4,a4,-48 # 3fc90fd0 <__l2_shared_end+0x23c7ce30>
1c0081aa:	02e40563          	beq	s0,a4,1c0081d4 <__ieee754_rem_pio2f+0x78>
1c0081ae:	1c001737          	lui	a4,0x1c001
1c0081b2:	bec72703          	lw	a4,-1044(a4) # 1c000bec <PIo2+0x290>
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
1c0081d8:	bf072703          	lw	a4,-1040(a4) # 1c000bf0 <PIo2+0x294>
1c0081dc:	08e7f7d3          	fsub.s	a5,a5,a4
1c0081e0:	1c001737          	lui	a4,0x1c001
1c0081e4:	bf472703          	lw	a4,-1036(a4) # 1c000bf4 <PIo2+0x298>
1c0081e8:	b7f9                	j	1c0081b6 <__ieee754_rem_pio2f+0x5a>
1c0081ea:	00e577d3          	fadd.s	a5,a0,a4
1c0081ee:	3fc91737          	lui	a4,0x3fc91
1c0081f2:	fd070713          	addi	a4,a4,-48 # 3fc90fd0 <__l2_shared_end+0x23c7ce30>
1c0081f6:	02e40063          	beq	s0,a4,1c008216 <__ieee754_rem_pio2f+0xba>
1c0081fa:	1c001737          	lui	a4,0x1c001
1c0081fe:	bec72703          	lw	a4,-1044(a4) # 1c000bec <PIo2+0x290>
1c008202:	00e7f6d3          	fadd.s	a3,a5,a4
1c008206:	557d                	li	a0,-1
1c008208:	08d7f7d3          	fsub.s	a5,a5,a3
1c00820c:	c194                	sw	a3,0(a1)
1c00820e:	00e7f7d3          	fadd.s	a5,a5,a4
1c008212:	c1dc                	sw	a5,4(a1)
1c008214:	bf55                	j	1c0081c8 <__ieee754_rem_pio2f+0x6c>
1c008216:	1c001737          	lui	a4,0x1c001
1c00821a:	bf072703          	lw	a4,-1040(a4) # 1c000bf0 <PIo2+0x294>
1c00821e:	00e7f7d3          	fadd.s	a5,a5,a4
1c008222:	1c001737          	lui	a4,0x1c001
1c008226:	bf472703          	lw	a4,-1036(a4) # 1c000bf4 <PIo2+0x298>
1c00822a:	bfe1                	j	1c008202 <__ieee754_rem_pio2f+0xa6>
1c00822c:	43491737          	lui	a4,0x43491
1c008230:	f8070713          	addi	a4,a4,-128 # 43490f80 <__l2_shared_end+0x2747cde0>
1c008234:	84ae                	mv	s1,a1
1c008236:	0e874f63          	blt	a4,s0,1c008334 <__ieee754_rem_pio2f+0x1d8>
1c00823a:	039000ef          	jal	ra,1c008a72 <fabsf>
1c00823e:	1c0016b7          	lui	a3,0x1c001
1c008242:	1c001737          	lui	a4,0x1c001
1c008246:	87aa                	mv	a5,a0
1c008248:	bf872703          	lw	a4,-1032(a4) # 1c000bf8 <PIo2+0x29c>
1c00824c:	bfc6a503          	lw	a0,-1028(a3) # 1c000bfc <PIo2+0x2a0>
1c008250:	46fd                	li	a3,31
1c008252:	50e7f743          	fmadd.s	a4,a5,a4,a0
1c008256:	c0071553          	fcvt.w.s	a0,a4,rtz
1c00825a:	1c001737          	lui	a4,0x1c001
1c00825e:	be872703          	lw	a4,-1048(a4) # 1c000be8 <PIo2+0x28c>
1c008262:	d0057653          	fcvt.s.w	a2,a0
1c008266:	78e677cb          	fnmsub.s	a5,a2,a4,a5
1c00826a:	1c001737          	lui	a4,0x1c001
1c00826e:	bec72703          	lw	a4,-1044(a4) # 1c000bec <PIo2+0x290>
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
1c0082d6:	bf072703          	lw	a4,-1040(a4) # 1c000bf0 <PIo2+0x294>
1c0082da:	78e676cb          	fnmsub.s	a3,a2,a4,a5
1c0082de:	08d7f7d3          	fsub.s	a5,a5,a3
1c0082e2:	78e6774b          	fnmsub.s	a4,a2,a4,a5
1c0082e6:	1c0017b7          	lui	a5,0x1c001
1c0082ea:	bf47a783          	lw	a5,-1036(a5) # 1c000bf4 <PIo2+0x298>
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
1c008310:	c007a703          	lw	a4,-1024(a5) # 1c000c00 <PIo2+0x2a4>
1c008314:	68e677cb          	fnmsub.s	a5,a2,a4,a3
1c008318:	08f6f6d3          	fsub.s	a3,a3,a5
1c00831c:	68e676cb          	fnmsub.s	a3,a2,a4,a3
1c008320:	1c001737          	lui	a4,0x1c001
1c008324:	c0472703          	lw	a4,-1020(a4) # 1c000c04 <PIo2+0x2a8>
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
1c00836c:	c087a783          	lw	a5,-1016(a5) # 1c000c08 <PIo2+0x2ac>
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
1c0083d6:	bfc72683          	lw	a3,-1028(a4) # 1c000bfc <PIo2+0x2a0>
1c0083da:	10b57553          	fmul.s	a0,a0,a1
1c0083de:	1c001737          	lui	a4,0x1c001
1c0083e2:	1c0015b7          	lui	a1,0x1c001
1c0083e6:	c145a583          	lw	a1,-1004(a1) # 1c000c14 <PIo2+0x2b8>
1c0083ea:	c1072703          	lw	a4,-1008(a4) # 1c000c10 <PIo2+0x2b4>
1c0083ee:	10d7f6d3          	fmul.s	a3,a5,a3
1c0083f2:	58e7f743          	fmadd.s	a4,a5,a4,a1
1c0083f6:	1c0015b7          	lui	a1,0x1c001
1c0083fa:	c185a583          	lw	a1,-1000(a1) # 1c000c18 <PIo2+0x2bc>
1c0083fe:	58f77743          	fmadd.s	a4,a4,a5,a1
1c008402:	1c0015b7          	lui	a1,0x1c001
1c008406:	c1c5a583          	lw	a1,-996(a1) # 1c000c1c <PIo2+0x2c0>
1c00840a:	58f77743          	fmadd.s	a4,a4,a5,a1
1c00840e:	1c0015b7          	lui	a1,0x1c001
1c008412:	c205a583          	lw	a1,-992(a1) # 1c000c20 <PIo2+0x2c4>
1c008416:	58f77743          	fmadd.s	a4,a4,a5,a1
1c00841a:	1c0015b7          	lui	a1,0x1c001
1c00841e:	c245a583          	lw	a1,-988(a1) # 1c000c24 <PIo2+0x2c8>
1c008422:	58f77743          	fmadd.s	a4,a4,a5,a1
1c008426:	10f77753          	fmul.s	a4,a4,a5
1c00842a:	50e7f7c7          	fmsub.s	a5,a5,a4,a0
1c00842e:	3e99a737          	lui	a4,0x3e99a
1c008432:	99970713          	addi	a4,a4,-1639 # 3e999999 <__l2_shared_end+0x229857f9>
1c008436:	00c74963          	blt	a4,a2,1c008448 <__kernel_cosf+0x90>
1c00843a:	08f6f7d3          	fsub.s	a5,a3,a5
1c00843e:	99c82503          	lw	a0,-1636(a6) # 1c00099c <PIo2+0x40>
1c008442:	08f57553          	fsub.s	a0,a0,a5
1c008446:	8082                	ret
1c008448:	3f480737          	lui	a4,0x3f480
1c00844c:	00c74e63          	blt	a4,a2,1c008468 <__kernel_cosf+0xb0>
1c008450:	ff000537          	lui	a0,0xff000
1c008454:	962a                	add	a2,a2,a0
1c008456:	99c82503          	lw	a0,-1636(a6)
1c00845a:	08c57553          	fsub.s	a0,a0,a2
1c00845e:	08c6f653          	fsub.s	a2,a3,a2
1c008462:	08f677d3          	fsub.s	a5,a2,a5
1c008466:	bff1                	j	1c008442 <__kernel_cosf+0x8a>
1c008468:	1c001737          	lui	a4,0x1c001
1c00846c:	c0c72603          	lw	a2,-1012(a4) # 1c000c0c <PIo2+0x2b0>
1c008470:	b7dd                	j	1c008456 <__kernel_cosf+0x9e>
1c008472:	99c82503          	lw	a0,-1636(a6)
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
1c008592:	c2c70713          	addi	a4,a4,-980 # 1c000c2c <PIo2+0x2d0>
1c008596:	4318                	lw	a4,0(a4)
1c008598:	842a                	mv	s0,a0
1c00859a:	10e57553          	fmul.s	a0,a0,a4
1c00859e:	29e9                	jal	1c008a78 <floorf>
1c0085a0:	1c001637          	lui	a2,0x1c001
1c0085a4:	c3060613          	addi	a2,a2,-976 # 1c000c30 <PIo2+0x2d4>
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
1c00861a:	99c5a503          	lw	a0,-1636(a1) # 1c00099c <PIo2+0x40>
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
1c008664:	ffc7a68b          	p.lw	a3,-4(a5!) # 31fffffc <__l2_shared_end+0x15febe5c>
1c008668:	18068363          	beqz	a3,1c0087ee <__kernel_rem_pio2f+0x376>
1c00866c:	1c0017b7          	lui	a5,0x1c001
1c008670:	99c7a503          	lw	a0,-1636(a5) # 1c00099c <PIo2+0x40>
1c008674:	85a6                	mv	a1,s1
1c008676:	2169                	jal	1c008b00 <scalbnf>
1c008678:	00241793          	slli	a5,s0,0x2
1c00867c:	1010                	addi	a2,sp,32
1c00867e:	00f606b3          	add	a3,a2,a5
1c008682:	1c001637          	lui	a2,0x1c001
1c008686:	c2862e03          	lw	t3,-984(a2) # 1c000c28 <PIo2+0x2cc>
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
1c0086c6:	c2860613          	addi	a2,a2,-984 # 1c000c28 <PIo2+0x2cc>
1c0086ca:	4210                	lw	a2,0(a2)
1c0086cc:	15fd                	addi	a1,a1,-1
1c0086ce:	10c57753          	fmul.s	a4,a0,a2
1c0086d2:	1c001637          	lui	a2,0x1c001
1c0086d6:	c0860613          	addi	a2,a2,-1016 # 1c000c08 <PIo2+0x2ac>
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
1c008714:	bfc72703          	lw	a4,-1028(a4) # 1c000bfc <PIo2+0x2a0>
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
1c008800:	c087a683          	lw	a3,-1016(a5) # 1c000c08 <PIo2+0x2ac>
1c008804:	a0a687d3          	fle.s	a5,a3,a0
1c008808:	cf9d                	beqz	a5,1c008846 <__kernel_rem_pio2f+0x3ce>
1c00880a:	1c0017b7          	lui	a5,0x1c001
1c00880e:	c287a783          	lw	a5,-984(a5) # 1c000c28 <PIo2+0x2cc>
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
1c008a04:	c3882803          	lw	a6,-968(a6) # 1c000c38 <PIo2+0x2dc>
1c008a08:	c347a783          	lw	a5,-972(a5) # 1c000c34 <PIo2+0x2d8>
1c008a0c:	10e576d3          	fmul.s	a3,a0,a4
1c008a10:	80f777c3          	fmadd.s	a5,a4,a5,a6
1c008a14:	1c001837          	lui	a6,0x1c001
1c008a18:	c3c82803          	lw	a6,-964(a6) # 1c000c3c <PIo2+0x2e0>
1c008a1c:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a20:	1c001837          	lui	a6,0x1c001
1c008a24:	c4082803          	lw	a6,-960(a6) # 1c000c40 <PIo2+0x2e4>
1c008a28:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a2c:	1c001837          	lui	a6,0x1c001
1c008a30:	c4482803          	lw	a6,-956(a6) # 1c000c44 <PIo2+0x2e8>
1c008a34:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a38:	ea11                	bnez	a2,1c008a4c <__kernel_sinf+0x66>
1c008a3a:	1c001637          	lui	a2,0x1c001
1c008a3e:	c4862583          	lw	a1,-952(a2) # 1c000c48 <PIo2+0x2ec>
1c008a42:	58f777c3          	fmadd.s	a5,a4,a5,a1
1c008a46:	50d7f543          	fmadd.s	a0,a5,a3,a0
1c008a4a:	8082                	ret
1c008a4c:	10f6f7d3          	fmul.s	a5,a3,a5
1c008a50:	1c001637          	lui	a2,0x1c001
1c008a54:	bfc62603          	lw	a2,-1028(a2) # 1c000bfc <PIo2+0x2a0>
1c008a58:	78c5f7c7          	fmsub.s	a5,a1,a2,a5
1c008a5c:	58e7f747          	fmsub.s	a4,a5,a4,a1
1c008a60:	1c0017b7          	lui	a5,0x1c001
1c008a64:	c4c7a783          	lw	a5,-948(a5) # 1c000c4c <PIo2+0x2f0>
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
1c008a94:	c506a683          	lw	a3,-944(a3) # 1c000c50 <PIo2+0x2f4>
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
1c008ab8:	fff60713          	addi	a4,a2,-1 # 7fffff <__l1_heap_size+0x7e4017>
1c008abc:	40d75733          	sra	a4,a4,a3
1c008ac0:	00a775b3          	and	a1,a4,a0
1c008ac4:	d5fd                	beqz	a1,1c008ab2 <floorf+0x3a>
1c008ac6:	1c0015b7          	lui	a1,0x1c001
1c008aca:	c505a583          	lw	a1,-944(a1) # 1c000c50 <PIo2+0x2f4>
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
1c008b20:	c547a783          	lw	a5,-940(a5) # 1c000c54 <PIo2+0x2f8>
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
1c008b52:	c507a403          	lw	s0,-944(a5) # 1c000c50 <PIo2+0x2f4>
1c008b56:	85aa                	mv	a1,a0
1c008b58:	8522                	mv	a0,s0
1c008b5a:	209d                	jal	1c008bc0 <copysignf>
1c008b5c:	10857553          	fmul.s	a0,a0,s0
1c008b60:	a00d                	j	1c008b82 <scalbnf+0x82>
1c008b62:	1c0017b7          	lui	a5,0x1c001
1c008b66:	c587a783          	lw	a5,-936(a5) # 1c000c58 <PIo2+0x2fc>
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
1c008b92:	35078793          	addi	a5,a5,848 # c350 <_l1_preload_size+0x8340>
1c008b96:	fab7cce3          	blt	a5,a1,1c008b4e <scalbnf+0x4e>
1c008b9a:	1c0017b7          	lui	a5,0x1c001
1c008b9e:	c587a403          	lw	s0,-936(a5) # 1c000c58 <PIo2+0x2fc>
1c008ba2:	bf55                	j	1c008b56 <scalbnf+0x56>
1c008ba4:	01978513          	addi	a0,a5,25
1c008ba8:	1c0017b7          	lui	a5,0x1c001
1c008bac:	c5c7a783          	lw	a5,-932(a5) # 1c000c5c <PIo2+0x300>
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
1c008c02:	371010ef          	jal	ra,1c00a772 <__rt_init>
1c008c06:	00000513          	li	a0,0
1c008c0a:	00000593          	li	a1,0
1c008c0e:	00001397          	auipc	t2,0x1
1c008c12:	90638393          	addi	t2,t2,-1786 # 1c009514 <main>
1c008c16:	000380e7          	jalr	t2
1c008c1a:	842a                	mv	s0,a0
1c008c1c:	4ff010ef          	jal	ra,1c00a91a <__rt_deinit>
1c008c20:	8522                	mv	a0,s0
1c008c22:	403020ef          	jal	ra,1c00b824 <exit>

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
1c008c86:	05460613          	addi	a2,a2,84 # 1c00acd6 <__rt_bridge_handle_notif>
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
1c008d84:	c2250513          	addi	a0,a0,-990 # 1c00a9a2 <__rt_handle_illegal_instr>
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
1c00904c:	1101                	addi	sp,sp,-32
1c00904e:	10001337          	lui	t1,0x10001
1c009052:	6685                	lui	a3,0x1
1c009054:	c84e                	sw	s3,16(sp)
1c009056:	ce22                	sw	s0,28(sp)
1c009058:	01830993          	addi	s3,t1,24 # 10001018 <twiddle_factors>
1c00905c:	cc26                	sw	s1,24(sp)
1c00905e:	ca4a                	sw	s2,20(sp)
1c009060:	c652                	sw	s4,12(sp)
1c009062:	c456                	sw	s5,8(sp)
1c009064:	c25a                	sw	s6,4(sp)
1c009066:	c05e                	sw	s7,0(sp)
1c009068:	96aa                	add	a3,a3,a0
1c00906a:	01830313          	addi	t1,t1,24
1c00906e:	882a                	mv	a6,a0
1c009070:	400950fb          	lp.setupi	x1,1024,1c009094 <fft_radix2+0x48>
1c009074:	00085703          	lhu	a4,0(a6)
1c009078:	0006d883          	lhu	a7,0(a3) # 1000 <__rt_hyper_pending_tasks_last+0xa98>
1c00907c:	00432e0b          	p.lw	t3,4(t1!)
1c009080:	0d1777d3          	fsub.h	a5,a4,a7
1c009084:	00f047d7          	pv.add.sc.h	a5,zero,a5
1c009088:	05177753          	fadd.h	a4,a4,a7
1c00908c:	87c7a7b3          	vfmul.h	a5,a5,t3
1c009090:	00e8122b          	p.sh	a4,4(a6!)
1c009094:	00f6a22b          	p.sw	a5,4(a3!)
1c009098:	1c0017b7          	lui	a5,0x1c001
1c00909c:	9887af83          	lw	t6,-1656(a5) # 1c000988 <PIo2+0x2c>
1c0090a0:	1c0017b7          	lui	a5,0x1c001
1c0090a4:	98c7af03          	lw	t5,-1652(a5) # 1c00098c <PIo2+0x30>
1c0090a8:	1c0017b7          	lui	a5,0x1c001
1c0090ac:	9907ae83          	lw	t4,-1648(a5) # 1c000990 <PIo2+0x34>
1c0090b0:	4aa5                	li	s5,9
1c0090b2:	4a09                	li	s4,2
1c0090b4:	20000413          	li	s0,512
1c0090b8:	07405163          	blez	s4,1c00911a <fft_radix2+0xce>
1c0090bc:	00341913          	slli	s2,s0,0x3
1c0090c0:	82aa                	mv	t0,a0
1c0090c2:	00241493          	slli	s1,s0,0x2
1c0090c6:	002a1b13          	slli	s6,s4,0x2
1c0090ca:	83d2                	mv	t2,s4
1c0090cc:	0263c07b          	lp.setup	x0,t2,1c009118 <fft_radix2+0xcc>
1c0090d0:	4785                	li	a5,1
1c0090d2:	005488b3          	add	a7,s1,t0
1c0090d6:	8e4e                	mv	t3,s3
1c0090d8:	8816                	mv	a6,t0
1c0090da:	04f46633          	p.max	a2,s0,a5
1c0090de:	01b640fb          	lp.setup	x1,a2,1c009114 <fft_radix2+0xc8>
1c0090e2:	0008a303          	lw	t1,0(a7)
1c0090e6:	00082683          	lw	a3,0(a6)
1c0090ea:	216e770b          	p.lw	a4,s6(t3!)
1c0090ee:	8466a7b3          	vfsub.h	a5,a3,t1
1c0090f2:	e877fbd7          	pv.shufflei1.sci.b	s7,a5,14
1c0090f6:	86e7a7b3          	vfmul.h	a5,a5,a4
1c0090fa:	8266a6b3          	vfadd.h	a3,a3,t1
1c0090fe:	87772733          	vfmul.h	a4,a4,s7
1c009102:	833e                	mv	t1,a5
1c009104:	c9f71357          	pv.shuffle2.b	t1,a4,t6
1c009108:	c9e717d7          	pv.shuffle2.b	a5,a4,t5
1c00910c:	00d8222b          	p.sw	a3,4(a6!)
1c009110:	906ea7b3          	vfmac.h	a5,t4,t1
1c009114:	00f8a22b          	p.sw	a5,4(a7!)
1c009118:	92ca                	add	t0,t0,s2
1c00911a:	1afd                	addi	s5,s5,-1
1c00911c:	8405                	srai	s0,s0,0x1
1c00911e:	0a06                	slli	s4,s4,0x1
1c009120:	f80a9ce3          	bnez	s5,1c0090b8 <fft_radix2+0x6c>
1c009124:	862e                	mv	a2,a1
1c009126:	00450313          	addi	t1,a0,4
1c00912a:	00458893          	addi	a7,a1,4
1c00912e:	882e                	mv	a6,a1
1c009130:	400650fb          	lp.setupi	x1,1024,1c009148 <fft_radix2+0xfc>
1c009134:	0085278b          	p.lw	a5,8(a0!)
1c009138:	0083268b          	p.lw	a3,8(t1!)
1c00913c:	82d7ae33          	vfadd.h	t3,a5,a3
1c009140:	01c8242b          	p.sw	t3,8(a6!)
1c009144:	84d7a7b3          	vfsub.h	a5,a5,a3
1c009148:	00f8a42b          	p.sw	a5,8(a7!)
1c00914c:	10000537          	lui	a0,0x10000
1c009150:	62c1                	lui	t0,0x10
1c009152:	01850513          	addi	a0,a0,24 # 10000018 <bit_rev_radix2_LUT>
1c009156:	4781                	li	a5,0
1c009158:	12fd                	addi	t0,t0,-1
1c00915a:	20000f13          	li	t5,512
1c00915e:	046f40fb          	lp.setup	x1,t5,1c0091ea <fft_radix2+0x19e>
1c009162:	4114                	lw	a3,0(a0)
1c009164:	4158                	lw	a4,4(a0)
1c009166:	00178493          	addi	s1,a5,1
1c00916a:	0056feb3          	and	t4,a3,t0
1c00916e:	002e9813          	slli	a6,t4,0x2
1c009172:	82c1                	srli	a3,a3,0x10
1c009174:	00577fb3          	and	t6,a4,t0
1c009178:	01058e33          	add	t3,a1,a6
1c00917c:	00269813          	slli	a6,a3,0x2
1c009180:	01058333          	add	t1,a1,a6
1c009184:	8341                	srli	a4,a4,0x10
1c009186:	002f9813          	slli	a6,t6,0x2
1c00918a:	010588b3          	add	a7,a1,a6
1c00918e:	00271813          	slli	a6,a4,0x2
1c009192:	00378393          	addi	t2,a5,3
1c009196:	00278413          	addi	s0,a5,2
1c00919a:	982e                	add	a6,a6,a1
1c00919c:	01d7fa63          	bleu	t4,a5,1c0091b0 <fft_radix2+0x164>
1c0091a0:	000e2903          	lw	s2,0(t3)
1c0091a4:	00062e83          	lw	t4,0(a2)
1c0091a8:	01262023          	sw	s2,0(a2)
1c0091ac:	01de2023          	sw	t4,0(t3)
1c0091b0:	00d4f963          	bleu	a3,s1,1c0091c2 <fft_radix2+0x176>
1c0091b4:	00032683          	lw	a3,0(t1)
1c0091b8:	00462e83          	lw	t4,4(a2)
1c0091bc:	c254                	sw	a3,4(a2)
1c0091be:	01d32023          	sw	t4,0(t1)
1c0091c2:	01f47963          	bleu	t6,s0,1c0091d4 <fft_radix2+0x188>
1c0091c6:	0008a683          	lw	a3,0(a7)
1c0091ca:	00862e83          	lw	t4,8(a2)
1c0091ce:	c614                	sw	a3,8(a2)
1c0091d0:	01d8a023          	sw	t4,0(a7)
1c0091d4:	00e3f963          	bleu	a4,t2,1c0091e6 <fft_radix2+0x19a>
1c0091d8:	00082703          	lw	a4,0(a6)
1c0091dc:	00c62e83          	lw	t4,12(a2)
1c0091e0:	c658                	sw	a4,12(a2)
1c0091e2:	01d82023          	sw	t4,0(a6)
1c0091e6:	0791                	addi	a5,a5,4
1c0091e8:	0521                	addi	a0,a0,8
1c0091ea:	0641                	addi	a2,a2,16
1c0091ec:	4472                	lw	s0,28(sp)
1c0091ee:	44e2                	lw	s1,24(sp)
1c0091f0:	4952                	lw	s2,20(sp)
1c0091f2:	49c2                	lw	s3,16(sp)
1c0091f4:	4a32                	lw	s4,12(sp)
1c0091f6:	4aa2                	lw	s5,8(sp)
1c0091f8:	4b12                	lw	s6,4(sp)
1c0091fa:	4b82                	lw	s7,0(sp)
1c0091fc:	6105                	addi	sp,sp,32
1c0091fe:	8082                	ret

1c009200 <compute_twiddles>:
1c009200:	1101                	addi	sp,sp,-32
1c009202:	1c0017b7          	lui	a5,0x1c001
1c009206:	c84a                	sw	s2,16(sp)
1c009208:	9987a503          	lw	a0,-1640(a5) # 1c000998 <PIo2+0x3c>
1c00920c:	10001937          	lui	s2,0x10001
1c009210:	1c0017b7          	lui	a5,0x1c001
1c009214:	ca26                	sw	s1,20(sp)
1c009216:	c452                	sw	s4,8(sp)
1c009218:	99c7a483          	lw	s1,-1636(a5) # 1c00099c <PIo2+0x40>
1c00921c:	01890913          	addi	s2,s2,24 # 10001018 <twiddle_factors>
1c009220:	1c0017b7          	lui	a5,0x1c001
1c009224:	6a05                	lui	s4,0x1
1c009226:	c64e                	sw	s3,12(sp)
1c009228:	c256                	sw	s5,4(sp)
1c00922a:	ce06                	sw	ra,28(sp)
1c00922c:	cc22                	sw	s0,24(sp)
1c00922e:	9947da83          	lhu	s5,-1644(a5) # 1c000994 <PIo2+0x38>
1c009232:	9a4a                	add	s4,s4,s2
1c009234:	4981                	li	s3,0
1c009236:	a831                	j	1c009252 <compute_twiddles+0x52>
1c009238:	d409f453          	fcvt.h.w	s0,s3
1c00923c:	15547453          	fmul.h	s0,s0,s5
1c009240:	40240453          	fcvt.s.h	s0,s0
1c009244:	8522                	mv	a0,s0
1c009246:	e5bfe0ef          	jal	ra,1c0080a0 <cosf>
1c00924a:	84aa                	mv	s1,a0
1c00924c:	8522                	mv	a0,s0
1c00924e:	eaffe0ef          	jal	ra,1c0080fc <sinf>
1c009252:	b0a4a4b3          	vfcpka.h.s	s1,s1,a0
1c009256:	0099222b          	p.sw	s1,4(s2!)
1c00925a:	0985                	addi	s3,s3,1
1c00925c:	fd491ee3          	bne	s2,s4,1c009238 <compute_twiddles+0x38>
1c009260:	40f2                	lw	ra,28(sp)
1c009262:	4462                	lw	s0,24(sp)
1c009264:	44d2                	lw	s1,20(sp)
1c009266:	4942                	lw	s2,16(sp)
1c009268:	49b2                	lw	s3,12(sp)
1c00926a:	4a22                	lw	s4,8(sp)
1c00926c:	4a92                	lw	s5,4(sp)
1c00926e:	6105                	addi	sp,sp,32
1c009270:	8082                	ret

1c009272 <end_of_call>:
1c009272:	1c0017b7          	lui	a5,0x1c001
1c009276:	4705                	li	a4,1
1c009278:	72e7ac23          	sw	a4,1848(a5) # 1c001738 <done>
1c00927c:	8082                	ret

1c00927e <cluster_entry>:
1c00927e:	4705                	li	a4,1
1c009280:	002047b7          	lui	a5,0x204
1c009284:	08e7a223          	sw	a4,132(a5) # 204084 <__l1_heap_size+0x1e809c>
1c009288:	20078693          	addi	a3,a5,512
1c00928c:	c298                	sw	a4,0(a3)
1c00928e:	20c78693          	addi	a3,a5,524
1c009292:	c298                	sw	a4,0(a3)
1c009294:	22078713          	addi	a4,a5,544
1c009298:	10100693          	li	a3,257
1c00929c:	c314                	sw	a3,0(a4)
1c00929e:	22c78793          	addi	a5,a5,556
1c0092a2:	c394                	sw	a3,0(a5)
1c0092a4:	1c0097b7          	lui	a5,0x1c009
1c0092a8:	2c278793          	addi	a5,a5,706 # 1c0092c2 <main_fn>
1c0092ac:	002046b7          	lui	a3,0x204
1c0092b0:	08f6a023          	sw	a5,128(a3) # 204080 <__l1_heap_size+0x1e8098>
1c0092b4:	002047b7          	lui	a5,0x204
1c0092b8:	0807a023          	sw	zero,128(a5) # 204080 <__l1_heap_size+0x1e8098>
1c0092bc:	01c76783          	p.elw	a5,28(a4) # 80001c <__l1_heap_size+0x7e4034>
1c0092c0:	8082                	ret

1c0092c2 <main_fn>:
1c0092c2:	7135                	addi	sp,sp,-160
1c0092c4:	f14027f3          	csrr	a5,mhartid
1c0092c8:	cf06                	sw	ra,156(sp)
1c0092ca:	cd22                	sw	s0,152(sp)
1c0092cc:	cb26                	sw	s1,148(sp)
1c0092ce:	c94a                	sw	s2,144(sp)
1c0092d0:	c74e                	sw	s3,140(sp)
1c0092d2:	c552                	sw	s4,136(sp)
1c0092d4:	c356                	sw	s5,132(sp)
1c0092d6:	c15a                	sw	s6,128(sp)
1c0092d8:	dede                	sw	s7,124(sp)
1c0092da:	dce2                	sw	s8,120(sp)
1c0092dc:	dae6                	sw	s9,116(sp)
1c0092de:	d8ea                	sw	s10,112(sp)
1c0092e0:	d6ee                	sw	s11,108(sp)
1c0092e2:	f457b7b3          	p.bclr	a5,a5,26,5
1c0092e6:	20078863          	beqz	a5,1c0094f6 <main_fn+0x234>
1c0092ea:	f14027f3          	csrr	a5,mhartid
1c0092ee:	477d                	li	a4,31
1c0092f0:	ca5797b3          	p.extractu	a5,a5,5,5
1c0092f4:	00e78863          	beq	a5,a4,1c009304 <main_fn+0x42>
1c0092f8:	002047b7          	lui	a5,0x204
1c0092fc:	20078793          	addi	a5,a5,512 # 204200 <__l1_heap_size+0x1e8218>
1c009300:	01c7e703          	p.elw	a4,28(a5)
1c009304:	f14027f3          	csrr	a5,mhartid
1c009308:	8795                	srai	a5,a5,0x5
1c00930a:	f267b933          	p.bclr	s2,a5,25,6
1c00930e:	4401                	li	s0,0
1c009310:	4981                	li	s3,0
1c009312:	4a01                	li	s4,0
1c009314:	4a81                	li	s5,0
1c009316:	4b01                	li	s6,0
1c009318:	4b81                	li	s7,0
1c00931a:	4c01                	li	s8,0
1c00931c:	4c81                	li	s9,0
1c00931e:	100024b7          	lui	s1,0x10002
1c009322:	8dca                	mv	s11,s2
1c009324:	8d4a                	mv	s10,s2
1c009326:	c24a                	sw	s2,4(sp)
1c009328:	c43e                	sw	a5,8(sp)
1c00932a:	c64a                	sw	s2,12(sp)
1c00932c:	a04d                	j	1c0093ce <main_fn+0x10c>
1c00932e:	102007b7          	lui	a5,0x10200
1c009332:	4705                	li	a4,1
1c009334:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fc3e8>
1c009338:	02e7a023          	sw	a4,32(a5)
1c00933c:	4781                	li	a5,0
1c00933e:	79f79073          	csrw	pccr31,a5
1c009342:	47fd                	li	a5,31
1c009344:	0afd8c63          	beq	s11,a5,1c0093fc <main_fn+0x13a>
1c009348:	102007b7          	lui	a5,0x10200
1c00934c:	4705                	li	a4,1
1c00934e:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fc3e8>
1c009352:	00e7ac23          	sw	a4,24(a5)
1c009356:	478d                	li	a5,3
1c009358:	cc179073          	csrw	0xcc1,a5
1c00935c:	01848593          	addi	a1,s1,24 # 10002018 <Input_Signal>
1c009360:	01848513          	addi	a0,s1,24
1c009364:	31e5                	jal	1c00904c <fft_radix2>
1c009366:	47fd                	li	a5,31
1c009368:	0afd0a63          	beq	s10,a5,1c00941c <main_fn+0x15a>
1c00936c:	102007b7          	lui	a5,0x10200
1c009370:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fc3e8>
1c009374:	0007a023          	sw	zero,0(a5)
1c009378:	4781                	li	a5,0
1c00937a:	cc179073          	csrw	0xcc1,a5
1c00937e:	0848                	addi	a0,sp,20
1c009380:	346010ef          	jal	ra,1c00a6c6 <rt_perf_save>
1c009384:	4785                	li	a5,1
1c009386:	0487d163          	ble	s0,a5,1c0093c8 <main_fn+0x106>
1c00938a:	4692                	lw	a3,4(sp)
1c00938c:	47fd                	li	a5,31
1c00938e:	08f68f63          	beq	a3,a5,1c00942c <main_fn+0x16a>
1c009392:	102007b7          	lui	a5,0x10200
1c009396:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fc3e8>
1c00939a:	0087a783          	lw	a5,8(a5)
1c00939e:	9cbe                	add	s9,s9,a5
1c0093a0:	78102773          	csrr	a4,pccr1
1c0093a4:	9c3a                	add	s8,s8,a4
1c0093a6:	780026f3          	csrr	a3,pccr0
1c0093aa:	9bb6                	add	s7,s7,a3
1c0093ac:	78c026f3          	csrr	a3,pccr12
1c0093b0:	9b36                	add	s6,s6,a3
1c0093b2:	790026f3          	csrr	a3,pccr16
1c0093b6:	9ab6                	add	s5,s5,a3
1c0093b8:	782026f3          	csrr	a3,pccr2
1c0093bc:	9a36                	add	s4,s4,a3
1c0093be:	784027f3          	csrr	a5,pccr4
1c0093c2:	99be                	add	s3,s3,a5
1c0093c4:	06642963          	p.beqimm	s0,6,1c009436 <main_fn+0x174>
1c0093c8:	0405                	addi	s0,s0,1
1c0093ca:	0e742a63          	p.beqimm	s0,7,1c0094be <main_fn+0x1fc>
1c0093ce:	0848                	addi	a0,sp,20
1c0093d0:	2e0010ef          	jal	ra,1c00a6b0 <rt_perf_init>
1c0093d4:	000315b7          	lui	a1,0x31
1c0093d8:	05dd                	addi	a1,a1,23
1c0093da:	0848                	addi	a0,sp,20
1c0093dc:	2e2010ef          	jal	ra,1c00a6be <rt_perf_conf>
1c0093e0:	47fd                	li	a5,31
1c0093e2:	f4f916e3          	bne	s2,a5,1c00932e <main_fn+0x6c>
1c0093e6:	4785                	li	a5,1
1c0093e8:	1a10b737          	lui	a4,0x1a10b
1c0093ec:	02f72023          	sw	a5,32(a4) # 1a10b020 <__l1_end+0xa107008>
1c0093f0:	4781                	li	a5,0
1c0093f2:	79f79073          	csrw	pccr31,a5
1c0093f6:	47fd                	li	a5,31
1c0093f8:	f4fd98e3          	bne	s11,a5,1c009348 <main_fn+0x86>
1c0093fc:	4785                	li	a5,1
1c0093fe:	1a10b737          	lui	a4,0x1a10b
1c009402:	00f72c23          	sw	a5,24(a4) # 1a10b018 <__l1_end+0xa107000>
1c009406:	478d                	li	a5,3
1c009408:	cc179073          	csrw	0xcc1,a5
1c00940c:	01848593          	addi	a1,s1,24
1c009410:	01848513          	addi	a0,s1,24
1c009414:	3925                	jal	1c00904c <fft_radix2>
1c009416:	47fd                	li	a5,31
1c009418:	f4fd1ae3          	bne	s10,a5,1c00936c <main_fn+0xaa>
1c00941c:	1a10b7b7          	lui	a5,0x1a10b
1c009420:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa106fe8>
1c009424:	4781                	li	a5,0
1c009426:	cc179073          	csrw	0xcc1,a5
1c00942a:	bf91                	j	1c00937e <main_fn+0xbc>
1c00942c:	1a10b7b7          	lui	a5,0x1a10b
1c009430:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa106ff0>
1c009434:	b7ad                	j	1c00939e <main_fn+0xdc>
1c009436:	4495                	li	s1,5
1c009438:	029cd633          	divu	a2,s9,s1
1c00943c:	f1402473          	csrr	s0,mhartid
1c009440:	1c001537          	lui	a0,0x1c001
1c009444:	f4543433          	p.bclr	s0,s0,26,5
1c009448:	85a2                	mv	a1,s0
1c00944a:	9a050513          	addi	a0,a0,-1632 # 1c0009a0 <PIo2+0x44>
1c00944e:	48e020ef          	jal	ra,1c00b8dc <printf>
1c009452:	029c5633          	divu	a2,s8,s1
1c009456:	1c001537          	lui	a0,0x1c001
1c00945a:	85a2                	mv	a1,s0
1c00945c:	9b450513          	addi	a0,a0,-1612 # 1c0009b4 <PIo2+0x58>
1c009460:	47c020ef          	jal	ra,1c00b8dc <printf>
1c009464:	029bd633          	divu	a2,s7,s1
1c009468:	1c001537          	lui	a0,0x1c001
1c00946c:	85a2                	mv	a1,s0
1c00946e:	9c850513          	addi	a0,a0,-1592 # 1c0009c8 <PIo2+0x6c>
1c009472:	46a020ef          	jal	ra,1c00b8dc <printf>
1c009476:	029b5633          	divu	a2,s6,s1
1c00947a:	1c001537          	lui	a0,0x1c001
1c00947e:	85a2                	mv	a1,s0
1c009480:	9e450513          	addi	a0,a0,-1564 # 1c0009e4 <PIo2+0x88>
1c009484:	458020ef          	jal	ra,1c00b8dc <printf>
1c009488:	029ad633          	divu	a2,s5,s1
1c00948c:	1c001537          	lui	a0,0x1c001
1c009490:	85a2                	mv	a1,s0
1c009492:	9fc50513          	addi	a0,a0,-1540 # 1c0009fc <PIo2+0xa0>
1c009496:	446020ef          	jal	ra,1c00b8dc <printf>
1c00949a:	029a5633          	divu	a2,s4,s1
1c00949e:	1c001537          	lui	a0,0x1c001
1c0094a2:	85a2                	mv	a1,s0
1c0094a4:	a1450513          	addi	a0,a0,-1516 # 1c000a14 <PIo2+0xb8>
1c0094a8:	434020ef          	jal	ra,1c00b8dc <printf>
1c0094ac:	0299d633          	divu	a2,s3,s1
1c0094b0:	1c001537          	lui	a0,0x1c001
1c0094b4:	85a2                	mv	a1,s0
1c0094b6:	a2c50513          	addi	a0,a0,-1492 # 1c000a2c <PIo2+0xd0>
1c0094ba:	422020ef          	jal	ra,1c00b8dc <printf>
1c0094be:	f14027f3          	csrr	a5,mhartid
1c0094c2:	477d                	li	a4,31
1c0094c4:	ca5797b3          	p.extractu	a5,a5,5,5
1c0094c8:	00e78863          	beq	a5,a4,1c0094d8 <main_fn+0x216>
1c0094cc:	002047b7          	lui	a5,0x204
1c0094d0:	20078793          	addi	a5,a5,512 # 204200 <__l1_heap_size+0x1e8218>
1c0094d4:	01c7e703          	p.elw	a4,28(a5)
1c0094d8:	40fa                	lw	ra,156(sp)
1c0094da:	446a                	lw	s0,152(sp)
1c0094dc:	44da                	lw	s1,148(sp)
1c0094de:	494a                	lw	s2,144(sp)
1c0094e0:	49ba                	lw	s3,140(sp)
1c0094e2:	4a2a                	lw	s4,136(sp)
1c0094e4:	4a9a                	lw	s5,132(sp)
1c0094e6:	4b0a                	lw	s6,128(sp)
1c0094e8:	5bf6                	lw	s7,124(sp)
1c0094ea:	5c66                	lw	s8,120(sp)
1c0094ec:	5cd6                	lw	s9,116(sp)
1c0094ee:	5d46                	lw	s10,112(sp)
1c0094f0:	5db6                	lw	s11,108(sp)
1c0094f2:	610d                	addi	sp,sp,160
1c0094f4:	8082                	ret
1c0094f6:	100004b7          	lui	s1,0x10000
1c0094fa:	3319                	jal	1c009200 <compute_twiddles>
1c0094fc:	01848493          	addi	s1,s1,24 # 10000018 <bit_rev_radix2_LUT>
1c009500:	4401                	li	s0,0
1c009502:	8522                	mv	a0,s0
1c009504:	3ce9                	jal	1c008fde <bit_rev_radix2>
1c009506:	0405                	addi	s0,s0,1
1c009508:	00a4912b          	p.sh	a0,2(s1!)
1c00950c:	80040793          	addi	a5,s0,-2048
1c009510:	fbed                	bnez	a5,1c009502 <main_fn+0x240>
1c009512:	bbe1                	j	1c0092ea <main_fn+0x28>

1c009514 <main>:
1c009514:	1101                	addi	sp,sp,-32
1c009516:	4591                	li	a1,4
1c009518:	e3ff7517          	auipc	a0,0xe3ff7
1c00951c:	af450513          	addi	a0,a0,-1292 # c <__rt_sched>
1c009520:	ce06                	sw	ra,28(sp)
1c009522:	cc22                	sw	s0,24(sp)
1c009524:	ca26                	sw	s1,20(sp)
1c009526:	c84a                	sw	s2,16(sp)
1c009528:	20c1                	jal	1c0095e8 <rt_event_alloc>
1c00952a:	e941                	bnez	a0,1c0095ba <main+0xa6>
1c00952c:	4681                	li	a3,0
1c00952e:	4601                	li	a2,0
1c009530:	4581                	li	a1,0
1c009532:	892a                	mv	s2,a0
1c009534:	4505                	li	a0,1
1c009536:	6cf000ef          	jal	ra,1c00a404 <rt_cluster_mount>
1c00953a:	6589                	lui	a1,0x2
1c00953c:	40058593          	addi	a1,a1,1024 # 2400 <__rt_hyper_pending_tasks_last+0x1e98>
1c009540:	450d                	li	a0,3
1c009542:	2935                	jal	1c00997e <rt_alloc>
1c009544:	842a                	mv	s0,a0
1c009546:	c935                	beqz	a0,1c0095ba <main+0xa6>
1c009548:	1c0095b7          	lui	a1,0x1c009
1c00954c:	4601                	li	a2,0
1c00954e:	27258593          	addi	a1,a1,626 # 1c009272 <end_of_call>
1c009552:	e3ff7517          	auipc	a0,0xe3ff7
1c009556:	aba50513          	addi	a0,a0,-1350 # c <__rt_sched>
1c00955a:	2211                	jal	1c00965e <rt_event_get>
1c00955c:	1c009637          	lui	a2,0x1c009
1c009560:	c02a                	sw	a0,0(sp)
1c009562:	40000793          	li	a5,1024
1c009566:	4881                	li	a7,0
1c009568:	40000813          	li	a6,1024
1c00956c:	8722                	mv	a4,s0
1c00956e:	4681                	li	a3,0
1c009570:	27e60613          	addi	a2,a2,638 # 1c00927e <cluster_entry>
1c009574:	4581                	li	a1,0
1c009576:	4501                	li	a0,0
1c009578:	1c0014b7          	lui	s1,0x1c001
1c00957c:	5f9000ef          	jal	ra,1c00a374 <rt_cluster_call>
1c009580:	73848493          	addi	s1,s1,1848 # 1c001738 <done>
1c009584:	409c                	lw	a5,0(s1)
1c009586:	ef89                	bnez	a5,1c0095a0 <main+0x8c>
1c009588:	30047473          	csrrci	s0,mstatus,8
1c00958c:	4585                	li	a1,1
1c00958e:	e3ff7517          	auipc	a0,0xe3ff7
1c009592:	a7e50513          	addi	a0,a0,-1410 # c <__rt_sched>
1c009596:	2225                	jal	1c0096be <__rt_event_execute>
1c009598:	30041073          	csrw	mstatus,s0
1c00959c:	409c                	lw	a5,0(s1)
1c00959e:	d7ed                	beqz	a5,1c009588 <main+0x74>
1c0095a0:	4681                	li	a3,0
1c0095a2:	4601                	li	a2,0
1c0095a4:	4581                	li	a1,0
1c0095a6:	4501                	li	a0,0
1c0095a8:	65d000ef          	jal	ra,1c00a404 <rt_cluster_mount>
1c0095ac:	40f2                	lw	ra,28(sp)
1c0095ae:	4462                	lw	s0,24(sp)
1c0095b0:	854a                	mv	a0,s2
1c0095b2:	44d2                	lw	s1,20(sp)
1c0095b4:	4942                	lw	s2,16(sp)
1c0095b6:	6105                	addi	sp,sp,32
1c0095b8:	8082                	ret
1c0095ba:	597d                	li	s2,-1
1c0095bc:	bfc5                	j	1c0095ac <main+0x98>

1c0095be <__rt_event_init>:
1c0095be:	02052023          	sw	zero,32(a0)
1c0095c2:	02052223          	sw	zero,36(a0)
1c0095c6:	02052823          	sw	zero,48(a0)
1c0095ca:	00052023          	sw	zero,0(a0)
1c0095ce:	8082                	ret

1c0095d0 <__rt_wait_event_prepare_blocking>:
1c0095d0:	00800793          	li	a5,8
1c0095d4:	4388                	lw	a0,0(a5)
1c0095d6:	4d18                	lw	a4,24(a0)
1c0095d8:	02052223          	sw	zero,36(a0)
1c0095dc:	c398                	sw	a4,0(a5)
1c0095de:	4785                	li	a5,1
1c0095e0:	d11c                	sw	a5,32(a0)
1c0095e2:	00052023          	sw	zero,0(a0)
1c0095e6:	8082                	ret

1c0095e8 <rt_event_alloc>:
1c0095e8:	1141                	addi	sp,sp,-16
1c0095ea:	c422                	sw	s0,8(sp)
1c0095ec:	842e                	mv	s0,a1
1c0095ee:	c606                	sw	ra,12(sp)
1c0095f0:	c226                	sw	s1,4(sp)
1c0095f2:	300474f3          	csrrci	s1,mstatus,8
1c0095f6:	f14027f3          	csrr	a5,mhartid
1c0095fa:	8795                	srai	a5,a5,0x5
1c0095fc:	f267b7b3          	p.bclr	a5,a5,25,6
1c009600:	477d                	li	a4,31
1c009602:	00378513          	addi	a0,a5,3
1c009606:	00e79363          	bne	a5,a4,1c00960c <rt_event_alloc+0x24>
1c00960a:	4501                	li	a0,0
1c00960c:	08c00593          	li	a1,140
1c009610:	02b405b3          	mul	a1,s0,a1
1c009614:	26ad                	jal	1c00997e <rt_alloc>
1c009616:	87aa                	mv	a5,a0
1c009618:	557d                	li	a0,-1
1c00961a:	cf91                	beqz	a5,1c009636 <rt_event_alloc+0x4e>
1c00961c:	00802683          	lw	a3,8(zero) # 8 <__rt_first_free>
1c009620:	4581                	li	a1,0
1c009622:	4601                	li	a2,0
1c009624:	00800713          	li	a4,8
1c009628:	00864c63          	blt	a2,s0,1c009640 <rt_event_alloc+0x58>
1c00962c:	c191                	beqz	a1,1c009630 <rt_event_alloc+0x48>
1c00962e:	c314                	sw	a3,0(a4)
1c009630:	30049073          	csrw	mstatus,s1
1c009634:	4501                	li	a0,0
1c009636:	40b2                	lw	ra,12(sp)
1c009638:	4422                	lw	s0,8(sp)
1c00963a:	4492                	lw	s1,4(sp)
1c00963c:	0141                	addi	sp,sp,16
1c00963e:	8082                	ret
1c009640:	cf94                	sw	a3,24(a5)
1c009642:	0207a023          	sw	zero,32(a5)
1c009646:	0207a223          	sw	zero,36(a5)
1c00964a:	0207a823          	sw	zero,48(a5)
1c00964e:	0007a023          	sw	zero,0(a5)
1c009652:	86be                	mv	a3,a5
1c009654:	0605                	addi	a2,a2,1
1c009656:	4585                	li	a1,1
1c009658:	08c78793          	addi	a5,a5,140
1c00965c:	b7f1                	j	1c009628 <rt_event_alloc+0x40>

1c00965e <rt_event_get>:
1c00965e:	30047773          	csrrci	a4,mstatus,8
1c009662:	00800793          	li	a5,8
1c009666:	4388                	lw	a0,0(a5)
1c009668:	c509                	beqz	a0,1c009672 <rt_event_get+0x14>
1c00966a:	4d14                	lw	a3,24(a0)
1c00966c:	c150                	sw	a2,4(a0)
1c00966e:	c394                	sw	a3,0(a5)
1c009670:	c10c                	sw	a1,0(a0)
1c009672:	30071073          	csrw	mstatus,a4
1c009676:	8082                	ret

1c009678 <rt_event_get_blocking>:
1c009678:	30047773          	csrrci	a4,mstatus,8
1c00967c:	00800793          	li	a5,8
1c009680:	4388                	lw	a0,0(a5)
1c009682:	c909                	beqz	a0,1c009694 <rt_event_get_blocking+0x1c>
1c009684:	4d14                	lw	a3,24(a0)
1c009686:	00052223          	sw	zero,4(a0)
1c00968a:	c394                	sw	a3,0(a5)
1c00968c:	4785                	li	a5,1
1c00968e:	00052023          	sw	zero,0(a0)
1c009692:	d11c                	sw	a5,32(a0)
1c009694:	30071073          	csrw	mstatus,a4
1c009698:	8082                	ret

1c00969a <rt_event_push>:
1c00969a:	30047773          	csrrci	a4,mstatus,8
1c00969e:	00800693          	li	a3,8
1c0096a2:	42d4                	lw	a3,4(a3)
1c0096a4:	00052c23          	sw	zero,24(a0)
1c0096a8:	00800793          	li	a5,8
1c0096ac:	e691                	bnez	a3,1c0096b8 <rt_event_push+0x1e>
1c0096ae:	c3c8                	sw	a0,4(a5)
1c0096b0:	c788                	sw	a0,8(a5)
1c0096b2:	30071073          	csrw	mstatus,a4
1c0096b6:	8082                	ret
1c0096b8:	4794                	lw	a3,8(a5)
1c0096ba:	ce88                	sw	a0,24(a3)
1c0096bc:	bfd5                	j	1c0096b0 <rt_event_push+0x16>

1c0096be <__rt_event_execute>:
1c0096be:	1141                	addi	sp,sp,-16
1c0096c0:	c422                	sw	s0,8(sp)
1c0096c2:	00800793          	li	a5,8
1c0096c6:	43dc                	lw	a5,4(a5)
1c0096c8:	c606                	sw	ra,12(sp)
1c0096ca:	c226                	sw	s1,4(sp)
1c0096cc:	00800413          	li	s0,8
1c0096d0:	eb91                	bnez	a5,1c0096e4 <__rt_event_execute+0x26>
1c0096d2:	c1a9                	beqz	a1,1c009714 <__rt_event_execute+0x56>
1c0096d4:	10500073          	wfi
1c0096d8:	30045073          	csrwi	mstatus,8
1c0096dc:	300477f3          	csrrci	a5,mstatus,8
1c0096e0:	405c                	lw	a5,4(s0)
1c0096e2:	cb8d                	beqz	a5,1c009714 <__rt_event_execute+0x56>
1c0096e4:	4485                	li	s1,1
1c0096e6:	4f98                	lw	a4,24(a5)
1c0096e8:	53d4                	lw	a3,36(a5)
1c0096ea:	00978823          	sb	s1,16(a5)
1c0096ee:	c058                	sw	a4,4(s0)
1c0096f0:	43c8                	lw	a0,4(a5)
1c0096f2:	4398                	lw	a4,0(a5)
1c0096f4:	e691                	bnez	a3,1c009700 <__rt_event_execute+0x42>
1c0096f6:	5394                	lw	a3,32(a5)
1c0096f8:	e681                	bnez	a3,1c009700 <__rt_event_execute+0x42>
1c0096fa:	4014                	lw	a3,0(s0)
1c0096fc:	c01c                	sw	a5,0(s0)
1c0096fe:	cf94                	sw	a3,24(a5)
1c009700:	0207a023          	sw	zero,32(a5)
1c009704:	c711                	beqz	a4,1c009710 <__rt_event_execute+0x52>
1c009706:	30045073          	csrwi	mstatus,8
1c00970a:	9702                	jalr	a4
1c00970c:	300477f3          	csrrci	a5,mstatus,8
1c009710:	405c                	lw	a5,4(s0)
1c009712:	fbf1                	bnez	a5,1c0096e6 <__rt_event_execute+0x28>
1c009714:	40b2                	lw	ra,12(sp)
1c009716:	4422                	lw	s0,8(sp)
1c009718:	4492                	lw	s1,4(sp)
1c00971a:	0141                	addi	sp,sp,16
1c00971c:	8082                	ret

1c00971e <__rt_wait_event>:
1c00971e:	1141                	addi	sp,sp,-16
1c009720:	c422                	sw	s0,8(sp)
1c009722:	c606                	sw	ra,12(sp)
1c009724:	842a                	mv	s0,a0
1c009726:	501c                	lw	a5,32(s0)
1c009728:	ef81                	bnez	a5,1c009740 <__rt_wait_event+0x22>
1c00972a:	581c                	lw	a5,48(s0)
1c00972c:	eb91                	bnez	a5,1c009740 <__rt_wait_event+0x22>
1c00972e:	00800793          	li	a5,8
1c009732:	4398                	lw	a4,0(a5)
1c009734:	40b2                	lw	ra,12(sp)
1c009736:	c380                	sw	s0,0(a5)
1c009738:	cc18                	sw	a4,24(s0)
1c00973a:	4422                	lw	s0,8(sp)
1c00973c:	0141                	addi	sp,sp,16
1c00973e:	8082                	ret
1c009740:	4585                	li	a1,1
1c009742:	4501                	li	a0,0
1c009744:	3fad                	jal	1c0096be <__rt_event_execute>
1c009746:	b7c5                	j	1c009726 <__rt_wait_event+0x8>

1c009748 <rt_event_wait>:
1c009748:	1141                	addi	sp,sp,-16
1c00974a:	c606                	sw	ra,12(sp)
1c00974c:	c422                	sw	s0,8(sp)
1c00974e:	30047473          	csrrci	s0,mstatus,8
1c009752:	37f1                	jal	1c00971e <__rt_wait_event>
1c009754:	30041073          	csrw	mstatus,s0
1c009758:	40b2                	lw	ra,12(sp)
1c00975a:	4422                	lw	s0,8(sp)
1c00975c:	0141                	addi	sp,sp,16
1c00975e:	8082                	ret

1c009760 <__rt_event_sched_init>:
1c009760:	00800513          	li	a0,8
1c009764:	00052023          	sw	zero,0(a0)
1c009768:	00052223          	sw	zero,4(a0)
1c00976c:	4585                	li	a1,1
1c00976e:	0511                	addi	a0,a0,4
1c009770:	bda5                	j	1c0095e8 <rt_event_alloc>

1c009772 <__rt_alloc_account>:
1c009772:	01052803          	lw	a6,16(a0)
1c009776:	495c                	lw	a5,20(a0)
1c009778:	4885                	li	a7,1
1c00977a:	010898b3          	sll	a7,a7,a6
1c00977e:	8d9d                	sub	a1,a1,a5
1c009780:	411007b3          	neg	a5,a7
1c009784:	8fed                	and	a5,a5,a1
1c009786:	40b885b3          	sub	a1,a7,a1
1c00978a:	0107d833          	srl	a6,a5,a6
1c00978e:	95be                	add	a1,a1,a5
1c009790:	04c5d5b3          	p.minu	a1,a1,a2
1c009794:	00281e93          	slli	t4,a6,0x2
1c009798:	4781                	li	a5,0
1c00979a:	4f05                	li	t5,1
1c00979c:	ea15                	bnez	a2,1c0097d0 <__rt_alloc_account+0x5e>
1c00979e:	c3dd                	beqz	a5,1c009844 <__rt_alloc_account+0xd2>
1c0097a0:	02402603          	lw	a2,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c0097a4:	c359                	beqz	a4,1c00982a <__rt_alloc_account+0xb8>
1c0097a6:	00479593          	slli	a1,a5,0x4
1c0097aa:	02802503          	lw	a0,40(zero) # 28 <__rt_alloc_l2_btrim_stdby>
1c0097ae:	07f6b263          	p.bneimm	a3,-1,1c009812 <__rt_alloc_account+0xa0>
1c0097b2:	8dc9                	or	a1,a1,a0
1c0097b4:	02b02423          	sw	a1,40(zero) # 28 <__rt_alloc_l2_btrim_stdby>
1c0097b8:	02802583          	lw	a1,40(zero) # 28 <__rt_alloc_l2_btrim_stdby>
1c0097bc:	1a104537          	lui	a0,0x1a104
1c0097c0:	16b52a23          	sw	a1,372(a0) # 1a104174 <__l1_end+0xa10015c>
1c0097c4:	07f6b563          	p.bneimm	a3,-1,1c00982e <__rt_alloc_account+0xbc>
1c0097c8:	fff7c793          	not	a5,a5
1c0097cc:	8e7d                	and	a2,a2,a5
1c0097ce:	a095                	j	1c009832 <__rt_alloc_account+0xc0>
1c0097d0:	cf15                	beqz	a4,1c00980c <__rt_alloc_account+0x9a>
1c0097d2:	00c52303          	lw	t1,12(a0)
1c0097d6:	9376                	add	t1,t1,t4
1c0097d8:	00032e03          	lw	t3,0(t1)
1c0097dc:	01f6b863          	p.bneimm	a3,-1,1c0097ec <__rt_alloc_account+0x7a>
1c0097e0:	01c89663          	bne	a7,t3,1c0097ec <__rt_alloc_account+0x7a>
1c0097e4:	010f1fb3          	sll	t6,t5,a6
1c0097e8:	01f7e7b3          	or	a5,a5,t6
1c0097ec:	42b68e33          	p.mac	t3,a3,a1
1c0097f0:	01c32023          	sw	t3,0(t1)
1c0097f4:	011e1663          	bne	t3,a7,1c009800 <__rt_alloc_account+0x8e>
1c0097f8:	010f1333          	sll	t1,t5,a6
1c0097fc:	0067e7b3          	or	a5,a5,t1
1c009800:	8e0d                	sub	a2,a2,a1
1c009802:	0805                	addi	a6,a6,1
1c009804:	0e91                	addi	t4,t4,4
1c009806:	04c8d5b3          	p.minu	a1,a7,a2
1c00980a:	bf49                	j	1c00979c <__rt_alloc_account+0x2a>
1c00980c:	00852303          	lw	t1,8(a0)
1c009810:	b7d9                	j	1c0097d6 <__rt_alloc_account+0x64>
1c009812:	fff5c593          	not	a1,a1
1c009816:	8de9                	and	a1,a1,a0
1c009818:	bf71                	j	1c0097b4 <__rt_alloc_account+0x42>
1c00981a:	07c2                	slli	a5,a5,0x10
1c00981c:	8e5d                	or	a2,a2,a5
1c00981e:	a811                	j	1c009832 <__rt_alloc_account+0xc0>
1c009820:	fff7c713          	not	a4,a5
1c009824:	8e79                	and	a2,a2,a4
1c009826:	07c2                	slli	a5,a5,0x10
1c009828:	b745                	j	1c0097c8 <__rt_alloc_account+0x56>
1c00982a:	fff6abe3          	p.beqimm	a3,-1,1c009820 <__rt_alloc_account+0xae>
1c00982e:	8e5d                	or	a2,a2,a5
1c009830:	d76d                	beqz	a4,1c00981a <__rt_alloc_account+0xa8>
1c009832:	02c02223          	sw	a2,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c009836:	02402783          	lw	a5,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c00983a:	1a104737          	lui	a4,0x1a104
1c00983e:	18f72023          	sw	a5,384(a4) # 1a104180 <__l1_end+0xa100168>
1c009842:	8082                	ret
1c009844:	8082                	ret

1c009846 <__rt_alloc_account_alloc>:
1c009846:	415c                	lw	a5,4(a0)
1c009848:	c781                	beqz	a5,1c009850 <__rt_alloc_account_alloc+0xa>
1c00984a:	4701                	li	a4,0
1c00984c:	56fd                	li	a3,-1
1c00984e:	b715                	j	1c009772 <__rt_alloc_account>
1c009850:	8082                	ret

1c009852 <__rt_alloc_account_free>:
1c009852:	415c                	lw	a5,4(a0)
1c009854:	c781                	beqz	a5,1c00985c <__rt_alloc_account_free+0xa>
1c009856:	4701                	li	a4,0
1c009858:	4685                	li	a3,1
1c00985a:	bf21                	j	1c009772 <__rt_alloc_account>
1c00985c:	8082                	ret

1c00985e <rt_user_alloc_init>:
1c00985e:	00758793          	addi	a5,a1,7
1c009862:	c407b7b3          	p.bclr	a5,a5,2,0
1c009866:	40b785b3          	sub	a1,a5,a1
1c00986a:	00052223          	sw	zero,4(a0)
1c00986e:	c11c                	sw	a5,0(a0)
1c009870:	8e0d                	sub	a2,a2,a1
1c009872:	00c05763          	blez	a2,1c009880 <rt_user_alloc_init+0x22>
1c009876:	c4063633          	p.bclr	a2,a2,2,0
1c00987a:	c390                	sw	a2,0(a5)
1c00987c:	0007a223          	sw	zero,4(a5)
1c009880:	8082                	ret

1c009882 <rt_user_alloc>:
1c009882:	1141                	addi	sp,sp,-16
1c009884:	c422                	sw	s0,8(sp)
1c009886:	4100                	lw	s0,0(a0)
1c009888:	059d                	addi	a1,a1,7
1c00988a:	c606                	sw	ra,12(sp)
1c00988c:	c226                	sw	s1,4(sp)
1c00988e:	c04a                	sw	s2,0(sp)
1c009890:	c405b7b3          	p.bclr	a5,a1,2,0
1c009894:	4701                	li	a4,0
1c009896:	cc19                	beqz	s0,1c0098b4 <rt_user_alloc+0x32>
1c009898:	4010                	lw	a2,0(s0)
1c00989a:	4054                	lw	a3,4(s0)
1c00989c:	02f64363          	blt	a2,a5,1c0098c2 <rt_user_alloc+0x40>
1c0098a0:	84aa                	mv	s1,a0
1c0098a2:	00840593          	addi	a1,s0,8
1c0098a6:	02f61363          	bne	a2,a5,1c0098cc <rt_user_alloc+0x4a>
1c0098aa:	cf19                	beqz	a4,1c0098c8 <rt_user_alloc+0x46>
1c0098ac:	c354                	sw	a3,4(a4)
1c0098ae:	1661                	addi	a2,a2,-8
1c0098b0:	8526                	mv	a0,s1
1c0098b2:	3f51                	jal	1c009846 <__rt_alloc_account_alloc>
1c0098b4:	8522                	mv	a0,s0
1c0098b6:	40b2                	lw	ra,12(sp)
1c0098b8:	4422                	lw	s0,8(sp)
1c0098ba:	4492                	lw	s1,4(sp)
1c0098bc:	4902                	lw	s2,0(sp)
1c0098be:	0141                	addi	sp,sp,16
1c0098c0:	8082                	ret
1c0098c2:	8722                	mv	a4,s0
1c0098c4:	8436                	mv	s0,a3
1c0098c6:	bfc1                	j	1c009896 <rt_user_alloc+0x14>
1c0098c8:	c094                	sw	a3,0(s1)
1c0098ca:	b7d5                	j	1c0098ae <rt_user_alloc+0x2c>
1c0098cc:	00f40933          	add	s2,s0,a5
1c0098d0:	8e1d                	sub	a2,a2,a5
1c0098d2:	00c92023          	sw	a2,0(s2)
1c0098d6:	00d92223          	sw	a3,4(s2)
1c0098da:	cb11                	beqz	a4,1c0098ee <rt_user_alloc+0x6c>
1c0098dc:	01272223          	sw	s2,4(a4)
1c0098e0:	ff878613          	addi	a2,a5,-8
1c0098e4:	8526                	mv	a0,s1
1c0098e6:	3785                	jal	1c009846 <__rt_alloc_account_alloc>
1c0098e8:	4621                	li	a2,8
1c0098ea:	85ca                	mv	a1,s2
1c0098ec:	b7d1                	j	1c0098b0 <rt_user_alloc+0x2e>
1c0098ee:	0124a023          	sw	s2,0(s1)
1c0098f2:	b7fd                	j	1c0098e0 <rt_user_alloc+0x5e>

1c0098f4 <rt_user_free>:
1c0098f4:	1101                	addi	sp,sp,-32
1c0098f6:	cc22                	sw	s0,24(sp)
1c0098f8:	842e                	mv	s0,a1
1c0098fa:	410c                	lw	a1,0(a0)
1c0098fc:	061d                	addi	a2,a2,7
1c0098fe:	ca26                	sw	s1,20(sp)
1c009900:	c84a                	sw	s2,16(sp)
1c009902:	c64e                	sw	s3,12(sp)
1c009904:	ce06                	sw	ra,28(sp)
1c009906:	89aa                	mv	s3,a0
1c009908:	c40634b3          	p.bclr	s1,a2,2,0
1c00990c:	4901                	li	s2,0
1c00990e:	c199                	beqz	a1,1c009914 <rt_user_free+0x20>
1c009910:	0485e763          	bltu	a1,s0,1c00995e <rt_user_free+0x6a>
1c009914:	009407b3          	add	a5,s0,s1
1c009918:	04f59663          	bne	a1,a5,1c009964 <rt_user_free+0x70>
1c00991c:	419c                	lw	a5,0(a1)
1c00991e:	4621                	li	a2,8
1c009920:	854e                	mv	a0,s3
1c009922:	97a6                	add	a5,a5,s1
1c009924:	c01c                	sw	a5,0(s0)
1c009926:	41dc                	lw	a5,4(a1)
1c009928:	c05c                	sw	a5,4(s0)
1c00992a:	3725                	jal	1c009852 <__rt_alloc_account_free>
1c00992c:	04090663          	beqz	s2,1c009978 <rt_user_free+0x84>
1c009930:	00092703          	lw	a4,0(s2)
1c009934:	00e907b3          	add	a5,s2,a4
1c009938:	02f41963          	bne	s0,a5,1c00996a <rt_user_free+0x76>
1c00993c:	401c                	lw	a5,0(s0)
1c00993e:	8626                	mv	a2,s1
1c009940:	85a2                	mv	a1,s0
1c009942:	97ba                	add	a5,a5,a4
1c009944:	00f92023          	sw	a5,0(s2)
1c009948:	405c                	lw	a5,4(s0)
1c00994a:	00f92223          	sw	a5,4(s2)
1c00994e:	4462                	lw	s0,24(sp)
1c009950:	40f2                	lw	ra,28(sp)
1c009952:	44d2                	lw	s1,20(sp)
1c009954:	4942                	lw	s2,16(sp)
1c009956:	854e                	mv	a0,s3
1c009958:	49b2                	lw	s3,12(sp)
1c00995a:	6105                	addi	sp,sp,32
1c00995c:	bddd                	j	1c009852 <__rt_alloc_account_free>
1c00995e:	892e                	mv	s2,a1
1c009960:	41cc                	lw	a1,4(a1)
1c009962:	b775                	j	1c00990e <rt_user_free+0x1a>
1c009964:	c004                	sw	s1,0(s0)
1c009966:	c04c                	sw	a1,4(s0)
1c009968:	b7d1                	j	1c00992c <rt_user_free+0x38>
1c00996a:	00892223          	sw	s0,4(s2)
1c00996e:	ff848613          	addi	a2,s1,-8
1c009972:	00840593          	addi	a1,s0,8
1c009976:	bfe1                	j	1c00994e <rt_user_free+0x5a>
1c009978:	0089a023          	sw	s0,0(s3)
1c00997c:	bfcd                	j	1c00996e <rt_user_free+0x7a>

1c00997e <rt_alloc>:
1c00997e:	1101                	addi	sp,sp,-32
1c009980:	cc22                	sw	s0,24(sp)
1c009982:	ce06                	sw	ra,28(sp)
1c009984:	ca26                	sw	s1,20(sp)
1c009986:	c84a                	sw	s2,16(sp)
1c009988:	c64e                	sw	s3,12(sp)
1c00998a:	c452                	sw	s4,8(sp)
1c00998c:	4789                	li	a5,2
1c00998e:	842a                	mv	s0,a0
1c009990:	02a7ed63          	bltu	a5,a0,1c0099ca <rt_alloc+0x4c>
1c009994:	1c001937          	lui	s2,0x1c001
1c009998:	89ae                	mv	s3,a1
1c00999a:	448d                	li	s1,3
1c00999c:	4a61                	li	s4,24
1c00999e:	76490913          	addi	s2,s2,1892 # 1c001764 <__rt_alloc_l2>
1c0099a2:	854a                	mv	a0,s2
1c0099a4:	43440533          	p.mac	a0,s0,s4
1c0099a8:	85ce                	mv	a1,s3
1c0099aa:	3de1                	jal	1c009882 <rt_user_alloc>
1c0099ac:	e519                	bnez	a0,1c0099ba <rt_alloc+0x3c>
1c0099ae:	0405                	addi	s0,s0,1
1c0099b0:	00343363          	p.bneimm	s0,3,1c0099b6 <rt_alloc+0x38>
1c0099b4:	4401                	li	s0,0
1c0099b6:	14fd                	addi	s1,s1,-1
1c0099b8:	f4ed                	bnez	s1,1c0099a2 <rt_alloc+0x24>
1c0099ba:	40f2                	lw	ra,28(sp)
1c0099bc:	4462                	lw	s0,24(sp)
1c0099be:	44d2                	lw	s1,20(sp)
1c0099c0:	4942                	lw	s2,16(sp)
1c0099c2:	49b2                	lw	s3,12(sp)
1c0099c4:	4a22                	lw	s4,8(sp)
1c0099c6:	6105                	addi	sp,sp,32
1c0099c8:	8082                	ret
1c0099ca:	1c0017b7          	lui	a5,0x1c001
1c0099ce:	ffd50413          	addi	s0,a0,-3
1c0099d2:	7607a503          	lw	a0,1888(a5) # 1c001760 <__rt_alloc_l1>
1c0099d6:	47e1                	li	a5,24
1c0099d8:	40f2                	lw	ra,28(sp)
1c0099da:	42f40533          	p.mac	a0,s0,a5
1c0099de:	4462                	lw	s0,24(sp)
1c0099e0:	44d2                	lw	s1,20(sp)
1c0099e2:	4942                	lw	s2,16(sp)
1c0099e4:	49b2                	lw	s3,12(sp)
1c0099e6:	4a22                	lw	s4,8(sp)
1c0099e8:	6105                	addi	sp,sp,32
1c0099ea:	bd61                	j	1c009882 <rt_user_alloc>

1c0099ec <__rt_alloc_init_l1>:
1c0099ec:	1c0017b7          	lui	a5,0x1c001
1c0099f0:	7607a703          	lw	a4,1888(a5) # 1c001760 <__rt_alloc_l1>
1c0099f4:	100047b7          	lui	a5,0x10004
1c0099f8:	01651593          	slli	a1,a0,0x16
1c0099fc:	01878793          	addi	a5,a5,24 # 10004018 <__l1_end>
1c009a00:	95be                	add	a1,a1,a5
1c009a02:	47e1                	li	a5,24
1c009a04:	42f50733          	p.mac	a4,a0,a5
1c009a08:	6671                	lui	a2,0x1c
1c009a0a:	fe860613          	addi	a2,a2,-24 # 1bfe8 <__l1_heap_size>
1c009a0e:	853a                	mv	a0,a4
1c009a10:	b5b9                	j	1c00985e <rt_user_alloc_init>

1c009a12 <__rt_alloc_init_l1_for_fc>:
1c009a12:	100045b7          	lui	a1,0x10004
1c009a16:	01651793          	slli	a5,a0,0x16
1c009a1a:	01858593          	addi	a1,a1,24 # 10004018 <__l1_end>
1c009a1e:	00b78733          	add	a4,a5,a1
1c009a22:	07e1                	addi	a5,a5,24
1c009a24:	1c0016b7          	lui	a3,0x1c001
1c009a28:	95be                	add	a1,a1,a5
1c009a2a:	47e1                	li	a5,24
1c009a2c:	76e6a023          	sw	a4,1888(a3) # 1c001760 <__rt_alloc_l1>
1c009a30:	42f50733          	p.mac	a4,a0,a5
1c009a34:	6671                	lui	a2,0x1c
1c009a36:	fd060613          	addi	a2,a2,-48 # 1bfd0 <_l1_preload_size+0x17fc0>
1c009a3a:	853a                	mv	a0,a4
1c009a3c:	b50d                	j	1c00985e <rt_user_alloc_init>

1c009a3e <__rt_allocs_init>:
1c009a3e:	1141                	addi	sp,sp,-16
1c009a40:	1c0015b7          	lui	a1,0x1c001
1c009a44:	c606                	sw	ra,12(sp)
1c009a46:	c422                	sw	s0,8(sp)
1c009a48:	c226                	sw	s1,4(sp)
1c009a4a:	c04a                	sw	s2,0(sp)
1c009a4c:	7f858793          	addi	a5,a1,2040 # 1c0017f8 <__l2_priv0_end>
1c009a50:	1c008637          	lui	a2,0x1c008
1c009a54:	0cc7c863          	blt	a5,a2,1c009b24 <__rt_allocs_init+0xe6>
1c009a58:	4581                	li	a1,0
1c009a5a:	4601                	li	a2,0
1c009a5c:	1c001437          	lui	s0,0x1c001
1c009a60:	76440513          	addi	a0,s0,1892 # 1c001764 <__rt_alloc_l2>
1c009a64:	3bed                	jal	1c00985e <rt_user_alloc_init>
1c009a66:	1c00c5b7          	lui	a1,0x1c00c
1c009a6a:	57058793          	addi	a5,a1,1392 # 1c00c570 <__l2_priv1_end>
1c009a6e:	1c010637          	lui	a2,0x1c010
1c009a72:	0ac7cd63          	blt	a5,a2,1c009b2c <__rt_allocs_init+0xee>
1c009a76:	4581                	li	a1,0
1c009a78:	4601                	li	a2,0
1c009a7a:	1c001537          	lui	a0,0x1c001
1c009a7e:	77c50513          	addi	a0,a0,1916 # 1c00177c <__rt_alloc_l2+0x18>
1c009a82:	3bf1                	jal	1c00985e <rt_user_alloc_init>
1c009a84:	1c0145b7          	lui	a1,0x1c014
1c009a88:	1a058793          	addi	a5,a1,416 # 1c0141a0 <__l2_shared_end>
1c009a8c:	1c190937          	lui	s2,0x1c190
1c009a90:	40f90933          	sub	s2,s2,a5
1c009a94:	1c0014b7          	lui	s1,0x1c001
1c009a98:	864a                	mv	a2,s2
1c009a9a:	1a058593          	addi	a1,a1,416
1c009a9e:	79448513          	addi	a0,s1,1940 # 1c001794 <__rt_alloc_l2+0x30>
1c009aa2:	3b75                	jal	1c00985e <rt_user_alloc_init>
1c009aa4:	76440413          	addi	s0,s0,1892
1c009aa8:	4785                	li	a5,1
1c009aaa:	d85c                	sw	a5,52(s0)
1c009aac:	03000593          	li	a1,48
1c009ab0:	4501                	li	a0,0
1c009ab2:	35f1                	jal	1c00997e <rt_alloc>
1c009ab4:	dc08                	sw	a0,56(s0)
1c009ab6:	03000593          	li	a1,48
1c009aba:	4501                	li	a0,0
1c009abc:	35c9                	jal	1c00997e <rt_alloc>
1c009abe:	5c14                	lw	a3,56(s0)
1c009ac0:	1c0017b7          	lui	a5,0x1c001
1c009ac4:	dc48                	sw	a0,60(s0)
1c009ac6:	76478793          	addi	a5,a5,1892 # 1c001764 <__rt_alloc_l2>
1c009aca:	00c250fb          	lp.setupi	x1,12,1c009ad2 <__rt_allocs_init+0x94>
1c009ace:	0006a22b          	p.sw	zero,4(a3!)
1c009ad2:	0005222b          	p.sw	zero,4(a0!)
1c009ad6:	4745                	li	a4,17
1c009ad8:	c3b8                	sw	a4,64(a5)
1c009ada:	1c0145b7          	lui	a1,0x1c014
1c009ade:	1c010737          	lui	a4,0x1c010
1c009ae2:	c3f8                	sw	a4,68(a5)
1c009ae4:	00890613          	addi	a2,s2,8 # 1c190008 <__l2_shared_end+0x17be68>
1c009ae8:	19858593          	addi	a1,a1,408 # 1c014198 <_l1_preload_start_inL2+0x4008>
1c009aec:	79448513          	addi	a0,s1,1940
1c009af0:	338d                	jal	1c009852 <__rt_alloc_account_free>
1c009af2:	f14027f3          	csrr	a5,mhartid
1c009af6:	ca5797b3          	p.extractu	a5,a5,5,5
1c009afa:	eb81                	bnez	a5,1c009b0a <__rt_allocs_init+0xcc>
1c009afc:	4422                	lw	s0,8(sp)
1c009afe:	40b2                	lw	ra,12(sp)
1c009b00:	4492                	lw	s1,4(sp)
1c009b02:	4902                	lw	s2,0(sp)
1c009b04:	4501                	li	a0,0
1c009b06:	0141                	addi	sp,sp,16
1c009b08:	b729                	j	1c009a12 <__rt_alloc_init_l1_for_fc>
1c009b0a:	45e1                	li	a1,24
1c009b0c:	4501                	li	a0,0
1c009b0e:	3d85                	jal	1c00997e <rt_alloc>
1c009b10:	40b2                	lw	ra,12(sp)
1c009b12:	4422                	lw	s0,8(sp)
1c009b14:	1c0017b7          	lui	a5,0x1c001
1c009b18:	76a7a023          	sw	a0,1888(a5) # 1c001760 <__rt_alloc_l1>
1c009b1c:	4492                	lw	s1,4(sp)
1c009b1e:	4902                	lw	s2,0(sp)
1c009b20:	0141                	addi	sp,sp,16
1c009b22:	8082                	ret
1c009b24:	8e1d                	sub	a2,a2,a5
1c009b26:	7f858593          	addi	a1,a1,2040
1c009b2a:	bf0d                	j	1c009a5c <__rt_allocs_init+0x1e>
1c009b2c:	8e1d                	sub	a2,a2,a5
1c009b2e:	57058593          	addi	a1,a1,1392
1c009b32:	b7a1                	j	1c009a7a <__rt_allocs_init+0x3c>

1c009b34 <__rt_time_poweroff>:
1c009b34:	1a10b7b7          	lui	a5,0x1a10b
1c009b38:	0791                	addi	a5,a5,4
1c009b3a:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa106ff0>
1c009b3e:	1c001737          	lui	a4,0x1c001
1c009b42:	72f72e23          	sw	a5,1852(a4) # 1c00173c <timer_count>
1c009b46:	4501                	li	a0,0
1c009b48:	8082                	ret

1c009b4a <__rt_time_poweron>:
1c009b4a:	1c0017b7          	lui	a5,0x1c001
1c009b4e:	73c7a703          	lw	a4,1852(a5) # 1c00173c <timer_count>
1c009b52:	1a10b7b7          	lui	a5,0x1a10b
1c009b56:	0791                	addi	a5,a5,4
1c009b58:	00e7a423          	sw	a4,8(a5) # 1a10b008 <__l1_end+0xa106ff0>
1c009b5c:	4501                	li	a0,0
1c009b5e:	8082                	ret

1c009b60 <rt_event_push_delayed>:
1c009b60:	30047373          	csrrci	t1,mstatus,8
1c009b64:	1c001637          	lui	a2,0x1c001
1c009b68:	7ac62703          	lw	a4,1964(a2) # 1c0017ac <first_delayed>
1c009b6c:	1a10b7b7          	lui	a5,0x1a10b
1c009b70:	0791                	addi	a5,a5,4
1c009b72:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa106ff0>
1c009b76:	46f9                	li	a3,30
1c009b78:	0405e5b3          	p.max	a1,a1,zero
1c009b7c:	02d5c5b3          	div	a1,a1,a3
1c009b80:	800006b7          	lui	a3,0x80000
1c009b84:	fff6c693          	not	a3,a3
1c009b88:	00d7f833          	and	a6,a5,a3
1c009b8c:	0585                	addi	a1,a1,1
1c009b8e:	97ae                	add	a5,a5,a1
1c009b90:	d95c                	sw	a5,52(a0)
1c009b92:	982e                	add	a6,a6,a1
1c009b94:	4781                	li	a5,0
1c009b96:	c719                	beqz	a4,1c009ba4 <rt_event_push_delayed+0x44>
1c009b98:	03472883          	lw	a7,52(a4)
1c009b9c:	00d8f8b3          	and	a7,a7,a3
1c009ba0:	0108e863          	bltu	a7,a6,1c009bb0 <rt_event_push_delayed+0x50>
1c009ba4:	cb89                	beqz	a5,1c009bb6 <rt_event_push_delayed+0x56>
1c009ba6:	cf88                	sw	a0,24(a5)
1c009ba8:	cd18                	sw	a4,24(a0)
1c009baa:	30031073          	csrw	mstatus,t1
1c009bae:	8082                	ret
1c009bb0:	87ba                	mv	a5,a4
1c009bb2:	4f18                	lw	a4,24(a4)
1c009bb4:	b7cd                	j	1c009b96 <rt_event_push_delayed+0x36>
1c009bb6:	1a10b7b7          	lui	a5,0x1a10b
1c009bba:	0791                	addi	a5,a5,4
1c009bbc:	7aa62623          	sw	a0,1964(a2)
1c009bc0:	cd18                	sw	a4,24(a0)
1c009bc2:	0087a703          	lw	a4,8(a5) # 1a10b008 <__l1_end+0xa106ff0>
1c009bc6:	95ba                	add	a1,a1,a4
1c009bc8:	00b7a823          	sw	a1,16(a5)
1c009bcc:	08500713          	li	a4,133
1c009bd0:	00e7a023          	sw	a4,0(a5)
1c009bd4:	bfd9                	j	1c009baa <rt_event_push_delayed+0x4a>

1c009bd6 <rt_time_wait_us>:
1c009bd6:	1101                	addi	sp,sp,-32
1c009bd8:	85aa                	mv	a1,a0
1c009bda:	4501                	li	a0,0
1c009bdc:	ce06                	sw	ra,28(sp)
1c009bde:	cc22                	sw	s0,24(sp)
1c009be0:	c62e                	sw	a1,12(sp)
1c009be2:	3c59                	jal	1c009678 <rt_event_get_blocking>
1c009be4:	45b2                	lw	a1,12(sp)
1c009be6:	842a                	mv	s0,a0
1c009be8:	3fa5                	jal	1c009b60 <rt_event_push_delayed>
1c009bea:	8522                	mv	a0,s0
1c009bec:	4462                	lw	s0,24(sp)
1c009bee:	40f2                	lw	ra,28(sp)
1c009bf0:	6105                	addi	sp,sp,32
1c009bf2:	be99                	j	1c009748 <rt_event_wait>

1c009bf4 <__rt_time_init>:
1c009bf4:	1c0017b7          	lui	a5,0x1c001
1c009bf8:	7a07a623          	sw	zero,1964(a5) # 1c0017ac <first_delayed>
1c009bfc:	1a10b7b7          	lui	a5,0x1a10b
1c009c00:	1141                	addi	sp,sp,-16
1c009c02:	08300713          	li	a4,131
1c009c06:	0791                	addi	a5,a5,4
1c009c08:	c606                	sw	ra,12(sp)
1c009c0a:	c422                	sw	s0,8(sp)
1c009c0c:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa106fe8>
1c009c10:	1c00a5b7          	lui	a1,0x1c00a
1c009c14:	c8e58593          	addi	a1,a1,-882 # 1c009c8e <__rt_timer_handler>
1c009c18:	452d                	li	a0,11
1c009c1a:	539000ef          	jal	ra,1c00a952 <rt_irq_set_handler>
1c009c1e:	6785                	lui	a5,0x1
1c009c20:	f1402773          	csrr	a4,mhartid
1c009c24:	46fd                	li	a3,31
1c009c26:	ca571733          	p.extractu	a4,a4,5,5
1c009c2a:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c009c2e:	04d71863          	bne	a4,a3,1c009c7e <__rt_time_init+0x8a>
1c009c32:	1a109737          	lui	a4,0x1a109
1c009c36:	c35c                	sw	a5,4(a4)
1c009c38:	1c00a5b7          	lui	a1,0x1c00a
1c009c3c:	4601                	li	a2,0
1c009c3e:	b3458593          	addi	a1,a1,-1228 # 1c009b34 <__rt_time_poweroff>
1c009c42:	4509                	li	a0,2
1c009c44:	69d000ef          	jal	ra,1c00aae0 <__rt_cbsys_add>
1c009c48:	1c00a5b7          	lui	a1,0x1c00a
1c009c4c:	842a                	mv	s0,a0
1c009c4e:	4601                	li	a2,0
1c009c50:	b4a58593          	addi	a1,a1,-1206 # 1c009b4a <__rt_time_poweron>
1c009c54:	450d                	li	a0,3
1c009c56:	68b000ef          	jal	ra,1c00aae0 <__rt_cbsys_add>
1c009c5a:	8d41                	or	a0,a0,s0
1c009c5c:	c50d                	beqz	a0,1c009c86 <__rt_time_init+0x92>
1c009c5e:	f1402673          	csrr	a2,mhartid
1c009c62:	1c001537          	lui	a0,0x1c001
1c009c66:	40565593          	srai	a1,a2,0x5
1c009c6a:	f265b5b3          	p.bclr	a1,a1,25,6
1c009c6e:	f4563633          	p.bclr	a2,a2,26,5
1c009c72:	a4050513          	addi	a0,a0,-1472 # 1c000a40 <PIo2+0xe4>
1c009c76:	467010ef          	jal	ra,1c00b8dc <printf>
1c009c7a:	3f1010ef          	jal	ra,1c00b86a <abort>
1c009c7e:	00204737          	lui	a4,0x204
1c009c82:	cb5c                	sw	a5,20(a4)
1c009c84:	bf55                	j	1c009c38 <__rt_time_init+0x44>
1c009c86:	40b2                	lw	ra,12(sp)
1c009c88:	4422                	lw	s0,8(sp)
1c009c8a:	0141                	addi	sp,sp,16
1c009c8c:	8082                	ret

1c009c8e <__rt_timer_handler>:
1c009c8e:	7179                	addi	sp,sp,-48
1c009c90:	ce36                	sw	a3,28(sp)
1c009c92:	1c0016b7          	lui	a3,0x1c001
1c009c96:	ca3e                	sw	a5,20(sp)
1c009c98:	7ac6a783          	lw	a5,1964(a3) # 1c0017ac <first_delayed>
1c009c9c:	cc3a                	sw	a4,24(sp)
1c009c9e:	1a10b737          	lui	a4,0x1a10b
1c009ca2:	0711                	addi	a4,a4,4
1c009ca4:	d61a                	sw	t1,44(sp)
1c009ca6:	d42a                	sw	a0,40(sp)
1c009ca8:	d22e                	sw	a1,36(sp)
1c009caa:	d032                	sw	a2,32(sp)
1c009cac:	c842                	sw	a6,16(sp)
1c009cae:	c646                	sw	a7,12(sp)
1c009cb0:	00872703          	lw	a4,8(a4) # 1a10b008 <__l1_end+0xa106ff0>
1c009cb4:	00c02583          	lw	a1,12(zero) # c <__rt_sched>
1c009cb8:	01002603          	lw	a2,16(zero) # 10 <__rt_sched+0x4>
1c009cbc:	800008b7          	lui	a7,0x80000
1c009cc0:	4501                	li	a0,0
1c009cc2:	4801                	li	a6,0
1c009cc4:	ffe8c893          	xori	a7,a7,-2
1c009cc8:	e3a5                	bnez	a5,1c009d28 <__rt_timer_handler+0x9a>
1c009cca:	00080463          	beqz	a6,1c009cd2 <__rt_timer_handler+0x44>
1c009cce:	00b02623          	sw	a1,12(zero) # c <__rt_sched>
1c009cd2:	c119                	beqz	a0,1c009cd8 <__rt_timer_handler+0x4a>
1c009cd4:	00c02823          	sw	a2,16(zero) # 10 <__rt_sched+0x4>
1c009cd8:	1a10b7b7          	lui	a5,0x1a10b
1c009cdc:	08100713          	li	a4,129
1c009ce0:	0791                	addi	a5,a5,4
1c009ce2:	7a06a623          	sw	zero,1964(a3)
1c009ce6:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa106fe8>
1c009cea:	6785                	lui	a5,0x1
1c009cec:	1a109737          	lui	a4,0x1a109
1c009cf0:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c009cf4:	cb5c                	sw	a5,20(a4)
1c009cf6:	5332                	lw	t1,44(sp)
1c009cf8:	5522                	lw	a0,40(sp)
1c009cfa:	5592                	lw	a1,36(sp)
1c009cfc:	5602                	lw	a2,32(sp)
1c009cfe:	46f2                	lw	a3,28(sp)
1c009d00:	4762                	lw	a4,24(sp)
1c009d02:	47d2                	lw	a5,20(sp)
1c009d04:	4842                	lw	a6,16(sp)
1c009d06:	48b2                	lw	a7,12(sp)
1c009d08:	6145                	addi	sp,sp,48
1c009d0a:	30200073          	mret
1c009d0e:	0187a303          	lw	t1,24(a5)
1c009d12:	0007ac23          	sw	zero,24(a5)
1c009d16:	c591                	beqz	a1,1c009d22 <__rt_timer_handler+0x94>
1c009d18:	ce1c                	sw	a5,24(a2)
1c009d1a:	863e                	mv	a2,a5
1c009d1c:	4505                	li	a0,1
1c009d1e:	879a                	mv	a5,t1
1c009d20:	b765                	j	1c009cc8 <__rt_timer_handler+0x3a>
1c009d22:	85be                	mv	a1,a5
1c009d24:	4805                	li	a6,1
1c009d26:	bfd5                	j	1c009d1a <__rt_timer_handler+0x8c>
1c009d28:	0347a303          	lw	t1,52(a5)
1c009d2c:	40670333          	sub	t1,a4,t1
1c009d30:	fc68ffe3          	bleu	t1,a7,1c009d0e <__rt_timer_handler+0x80>
1c009d34:	00080463          	beqz	a6,1c009d3c <__rt_timer_handler+0xae>
1c009d38:	00b02623          	sw	a1,12(zero) # c <__rt_sched>
1c009d3c:	c119                	beqz	a0,1c009d42 <__rt_timer_handler+0xb4>
1c009d3e:	00c02823          	sw	a2,16(zero) # 10 <__rt_sched+0x4>
1c009d42:	7af6a623          	sw	a5,1964(a3)
1c009d46:	1a10b6b7          	lui	a3,0x1a10b
1c009d4a:	0691                	addi	a3,a3,4
1c009d4c:	0086a603          	lw	a2,8(a3) # 1a10b008 <__l1_end+0xa106ff0>
1c009d50:	5bdc                	lw	a5,52(a5)
1c009d52:	40e78733          	sub	a4,a5,a4
1c009d56:	9732                	add	a4,a4,a2
1c009d58:	00e6a823          	sw	a4,16(a3)
1c009d5c:	08500793          	li	a5,133
1c009d60:	00f6a023          	sw	a5,0(a3)
1c009d64:	bf49                	j	1c009cf6 <__rt_timer_handler+0x68>

1c009d66 <__rt_pmu_change_domain_power>:
1c009d66:	1c0017b7          	lui	a5,0x1c001
1c009d6a:	4607a883          	lw	a7,1120(a5) # 1c001460 <stack>
1c009d6e:	ffd60813          	addi	a6,a2,-3
1c009d72:	46078793          	addi	a5,a5,1120
1c009d76:	06089563          	bnez	a7,1c009de0 <__rt_pmu_change_domain_power+0x7a>
1c009d7a:	4585                	li	a1,1
1c009d7c:	0105f463          	bleu	a6,a1,1c009d84 <__rt_pmu_change_domain_power+0x1e>
1c009d80:	02a02023          	sw	a0,32(zero) # 20 <__rt_pmu_scu_event>
1c009d84:	0047a803          	lw	a6,4(a5)
1c009d88:	4505                	li	a0,1
1c009d8a:	00c515b3          	sll	a1,a0,a2
1c009d8e:	fff5c593          	not	a1,a1
1c009d92:	0105f5b3          	and	a1,a1,a6
1c009d96:	00c69833          	sll	a6,a3,a2
1c009d9a:	0105e5b3          	or	a1,a1,a6
1c009d9e:	c3cc                	sw	a1,4(a5)
1c009da0:	c388                	sw	a0,0(a5)
1c009da2:	02463b63          	p.bneimm	a2,4,1c009dd8 <__rt_pmu_change_domain_power+0x72>
1c009da6:	fc1737b3          	p.bclr	a5,a4,30,1
1c009daa:	8b09                	andi	a4,a4,2
1c009dac:	00478693          	addi	a3,a5,4
1c009db0:	c319                	beqz	a4,1c009db6 <__rt_pmu_change_domain_power+0x50>
1c009db2:	00e78693          	addi	a3,a5,14
1c009db6:	0036d793          	srli	a5,a3,0x3
1c009dba:	6741                	lui	a4,0x10
1c009dbc:	0789                	addi	a5,a5,2
1c009dbe:	f836b6b3          	p.bclr	a3,a3,28,3
1c009dc2:	0786                	slli	a5,a5,0x1
1c009dc4:	00d716b3          	sll	a3,a4,a3
1c009dc8:	8edd                	or	a3,a3,a5
1c009dca:	0416e693          	ori	a3,a3,65
1c009dce:	1a1077b7          	lui	a5,0x1a107
1c009dd2:	00d7a023          	sw	a3,0(a5) # 1a107000 <__l1_end+0xa102fe8>
1c009dd6:	8082                	ret
1c009dd8:	060d                	addi	a2,a2,3
1c009dda:	0606                	slli	a2,a2,0x1
1c009ddc:	96b2                	add	a3,a3,a2
1c009dde:	bfe1                	j	1c009db6 <__rt_pmu_change_domain_power+0x50>
1c009de0:	dd58                	sw	a4,60(a0)
1c009de2:	00283813          	sltiu	a6,a6,2
1c009de6:	4798                	lw	a4,8(a5)
1c009de8:	00184813          	xori	a6,a6,1
1c009dec:	d950                	sw	a2,52(a0)
1c009dee:	dd14                	sw	a3,56(a0)
1c009df0:	05052023          	sw	a6,64(a0)
1c009df4:	eb09                	bnez	a4,1c009e06 <__rt_pmu_change_domain_power+0xa0>
1c009df6:	c788                	sw	a0,8(a5)
1c009df8:	c7c8                	sw	a0,12(a5)
1c009dfa:	00052c23          	sw	zero,24(a0)
1c009dfe:	c199                	beqz	a1,1c009e04 <__rt_pmu_change_domain_power+0x9e>
1c009e00:	4785                	li	a5,1
1c009e02:	c19c                	sw	a5,0(a1)
1c009e04:	8082                	ret
1c009e06:	47d8                	lw	a4,12(a5)
1c009e08:	cf08                	sw	a0,24(a4)
1c009e0a:	b7fd                	j	1c009df8 <__rt_pmu_change_domain_power+0x92>

1c009e0c <__rt_pmu_cluster_power_down>:
1c009e0c:	4701                	li	a4,0
1c009e0e:	4681                	li	a3,0
1c009e10:	460d                	li	a2,3
1c009e12:	bf91                	j	1c009d66 <__rt_pmu_change_domain_power>

1c009e14 <__rt_pmu_cluster_power_up>:
1c009e14:	1141                	addi	sp,sp,-16
1c009e16:	4701                	li	a4,0
1c009e18:	4685                	li	a3,1
1c009e1a:	460d                	li	a2,3
1c009e1c:	c606                	sw	ra,12(sp)
1c009e1e:	37a1                	jal	1c009d66 <__rt_pmu_change_domain_power>
1c009e20:	40b2                	lw	ra,12(sp)
1c009e22:	02002223          	sw	zero,36(zero) # 24 <__rt_alloc_l2_pwr_ctrl>
1c009e26:	03000713          	li	a4,48
1c009e2a:	1c0017b7          	lui	a5,0x1c001
1c009e2e:	46e7a823          	sw	a4,1136(a5) # 1c001470 <__rt_alloc_l1_pwr_ctrl>
1c009e32:	4505                	li	a0,1
1c009e34:	0141                	addi	sp,sp,16
1c009e36:	8082                	ret

1c009e38 <__rt_pmu_init>:
1c009e38:	1141                	addi	sp,sp,-16
1c009e3a:	1c0017b7          	lui	a5,0x1c001
1c009e3e:	1c00a5b7          	lui	a1,0x1c00a
1c009e42:	46078793          	addi	a5,a5,1120 # 1c001460 <stack>
1c009e46:	c606                	sw	ra,12(sp)
1c009e48:	4741                	li	a4,16
1c009e4a:	eaa58593          	addi	a1,a1,-342 # 1c009eaa <__rt_pmu_scu_handler>
1c009e4e:	4565                	li	a0,25
1c009e50:	c3d8                	sw	a4,4(a5)
1c009e52:	0007a023          	sw	zero,0(a5)
1c009e56:	0007a423          	sw	zero,8(a5)
1c009e5a:	0007aa23          	sw	zero,20(a5)
1c009e5e:	2f5000ef          	jal	ra,1c00a952 <rt_irq_set_handler>
1c009e62:	477d                	li	a4,31
1c009e64:	f14027f3          	csrr	a5,mhartid
1c009e68:	ca5797b3          	p.extractu	a5,a5,5,5
1c009e6c:	02e79963          	bne	a5,a4,1c009e9e <__rt_pmu_init+0x66>
1c009e70:	1a1097b7          	lui	a5,0x1a109
1c009e74:	02000737          	lui	a4,0x2000
1c009e78:	c3d8                	sw	a4,4(a5)
1c009e7a:	479d                	li	a5,7
1c009e7c:	1a107737          	lui	a4,0x1a107
1c009e80:	00f72623          	sw	a5,12(a4) # 1a10700c <__l1_end+0xa102ff4>
1c009e84:	40b2                	lw	ra,12(sp)
1c009e86:	00100737          	lui	a4,0x100
1c009e8a:	02000793          	li	a5,32
1c009e8e:	1741                	addi	a4,a4,-16
1c009e90:	0007a023          	sw	zero,0(a5) # 1a109000 <__l1_end+0xa104fe8>
1c009e94:	c798                	sw	a4,8(a5)
1c009e96:	0007a223          	sw	zero,4(a5)
1c009e9a:	0141                	addi	sp,sp,16
1c009e9c:	8082                	ret
1c009e9e:	002047b7          	lui	a5,0x204
1c009ea2:	02000737          	lui	a4,0x2000
1c009ea6:	cbd8                	sw	a4,20(a5)
1c009ea8:	bfc9                	j	1c009e7a <__rt_pmu_init+0x42>

1c009eaa <__rt_pmu_scu_handler>:
1c009eaa:	7179                	addi	sp,sp,-48
1c009eac:	cc3a                	sw	a4,24(sp)
1c009eae:	ca3e                	sw	a5,20(sp)
1c009eb0:	1a107737          	lui	a4,0x1a107
1c009eb4:	47c1                	li	a5,16
1c009eb6:	d61a                	sw	t1,44(sp)
1c009eb8:	d42a                	sw	a0,40(sp)
1c009eba:	d22e                	sw	a1,36(sp)
1c009ebc:	d032                	sw	a2,32(sp)
1c009ebe:	ce36                	sw	a3,28(sp)
1c009ec0:	c842                	sw	a6,16(sp)
1c009ec2:	c646                	sw	a7,12(sp)
1c009ec4:	00f72823          	sw	a5,16(a4) # 1a107010 <__l1_end+0xa102ff8>
1c009ec8:	02002783          	lw	a5,32(zero) # 20 <__rt_pmu_scu_event>
1c009ecc:	1c001737          	lui	a4,0x1c001
1c009ed0:	46072023          	sw	zero,1120(a4) # 1c001460 <stack>
1c009ed4:	853a                	mv	a0,a4
1c009ed6:	cf81                	beqz	a5,1c009eee <__rt_pmu_scu_handler+0x44>
1c009ed8:	00c02703          	lw	a4,12(zero) # c <__rt_sched>
1c009edc:	0007ac23          	sw	zero,24(a5) # 204018 <__l1_heap_size+0x1e8030>
1c009ee0:	e345                	bnez	a4,1c009f80 <__rt_pmu_scu_handler+0xd6>
1c009ee2:	00f02623          	sw	a5,12(zero) # c <__rt_sched>
1c009ee6:	00f02823          	sw	a5,16(zero) # 10 <__rt_sched+0x4>
1c009eea:	02002023          	sw	zero,32(zero) # 20 <__rt_pmu_scu_event>
1c009eee:	1c0017b7          	lui	a5,0x1c001
1c009ef2:	4687a683          	lw	a3,1128(a5) # 1c001468 <__rt_pmu_pending_requests>
1c009ef6:	caad                	beqz	a3,1c009f68 <__rt_pmu_scu_handler+0xbe>
1c009ef8:	4e98                	lw	a4,24(a3)
1c009efa:	1c0018b7          	lui	a7,0x1c001
1c009efe:	4648a303          	lw	t1,1124(a7) # 1c001464 <__rt_pmu_domains_on>
1c009f02:	46e7a423          	sw	a4,1128(a5)
1c009f06:	5ad8                	lw	a4,52(a3)
1c009f08:	4785                	li	a5,1
1c009f0a:	0386a803          	lw	a6,56(a3)
1c009f0e:	00e79633          	sll	a2,a5,a4
1c009f12:	fff64613          	not	a2,a2
1c009f16:	00667633          	and	a2,a2,t1
1c009f1a:	00e81333          	sll	t1,a6,a4
1c009f1e:	00666633          	or	a2,a2,t1
1c009f22:	46c8a223          	sw	a2,1124(a7)
1c009f26:	46f52023          	sw	a5,1120(a0)
1c009f2a:	5ecc                	lw	a1,60(a3)
1c009f2c:	04473e63          	p.bneimm	a4,4,1c009f88 <__rt_pmu_scu_handler+0xde>
1c009f30:	fc15b733          	p.bclr	a4,a1,30,1
1c009f34:	8989                	andi	a1,a1,2
1c009f36:	00470793          	addi	a5,a4,4
1c009f3a:	c199                	beqz	a1,1c009f40 <__rt_pmu_scu_handler+0x96>
1c009f3c:	00e70793          	addi	a5,a4,14
1c009f40:	0037d713          	srli	a4,a5,0x3
1c009f44:	6641                	lui	a2,0x10
1c009f46:	0709                	addi	a4,a4,2
1c009f48:	f837b7b3          	p.bclr	a5,a5,28,3
1c009f4c:	0706                	slli	a4,a4,0x1
1c009f4e:	00f617b3          	sll	a5,a2,a5
1c009f52:	8fd9                	or	a5,a5,a4
1c009f54:	0417e793          	ori	a5,a5,65
1c009f58:	1a107737          	lui	a4,0x1a107
1c009f5c:	00f72023          	sw	a5,0(a4) # 1a107000 <__l1_end+0xa102fe8>
1c009f60:	42bc                	lw	a5,64(a3)
1c009f62:	cb85                	beqz	a5,1c009f92 <__rt_pmu_scu_handler+0xe8>
1c009f64:	02d02023          	sw	a3,32(zero) # 20 <__rt_pmu_scu_event>
1c009f68:	5332                	lw	t1,44(sp)
1c009f6a:	5522                	lw	a0,40(sp)
1c009f6c:	5592                	lw	a1,36(sp)
1c009f6e:	5602                	lw	a2,32(sp)
1c009f70:	46f2                	lw	a3,28(sp)
1c009f72:	4762                	lw	a4,24(sp)
1c009f74:	47d2                	lw	a5,20(sp)
1c009f76:	4842                	lw	a6,16(sp)
1c009f78:	48b2                	lw	a7,12(sp)
1c009f7a:	6145                	addi	sp,sp,48
1c009f7c:	30200073          	mret
1c009f80:	01002703          	lw	a4,16(zero) # 10 <__rt_sched+0x4>
1c009f84:	cf1c                	sw	a5,24(a4)
1c009f86:	b785                	j	1c009ee6 <__rt_pmu_scu_handler+0x3c>
1c009f88:	00370793          	addi	a5,a4,3
1c009f8c:	0786                	slli	a5,a5,0x1
1c009f8e:	97c2                	add	a5,a5,a6
1c009f90:	bf45                	j	1c009f40 <__rt_pmu_scu_handler+0x96>
1c009f92:	00c02783          	lw	a5,12(zero) # c <__rt_sched>
1c009f96:	0006ac23          	sw	zero,24(a3)
1c009f9a:	e791                	bnez	a5,1c009fa6 <__rt_pmu_scu_handler+0xfc>
1c009f9c:	00d02623          	sw	a3,12(zero) # c <__rt_sched>
1c009fa0:	00d02823          	sw	a3,16(zero) # 10 <__rt_sched+0x4>
1c009fa4:	b7d1                	j	1c009f68 <__rt_pmu_scu_handler+0xbe>
1c009fa6:	01002783          	lw	a5,16(zero) # 10 <__rt_sched+0x4>
1c009faa:	cf94                	sw	a3,24(a5)
1c009fac:	bfd5                	j	1c009fa0 <__rt_pmu_scu_handler+0xf6>

1c009fae <__rt_init_cluster_data>:
1c009fae:	04050713          	addi	a4,a0,64
1c009fb2:	00800793          	li	a5,8
1c009fb6:	01671613          	slli	a2,a4,0x16
1c009fba:	e6c7b7b3          	p.bclr	a5,a5,19,12
1c009fbe:	1c0106b7          	lui	a3,0x1c010
1c009fc2:	97b2                	add	a5,a5,a2
1c009fc4:	6711                	lui	a4,0x4
1c009fc6:	19068693          	addi	a3,a3,400 # 1c010190 <_l1_preload_start_inL2>
1c009fca:	01070713          	addi	a4,a4,16 # 4010 <_l1_preload_size>
1c009fce:	8f95                	sub	a5,a5,a3
1c009fd0:	00f685b3          	add	a1,a3,a5
1c009fd4:	02e04963          	bgtz	a4,1c00a006 <__rt_init_cluster_data+0x58>
1c009fd8:	1c0017b7          	lui	a5,0x1c001
1c009fdc:	02800713          	li	a4,40
1c009fe0:	7b478793          	addi	a5,a5,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c009fe4:	42e507b3          	p.mac	a5,a0,a4
1c009fe8:	00201737          	lui	a4,0x201
1c009fec:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e4e1c>
1c009ff0:	9732                	add	a4,a4,a2
1c009ff2:	cb98                	sw	a4,16(a5)
1c009ff4:	00800713          	li	a4,8
1c009ff8:	e6c73733          	p.bclr	a4,a4,19,12
1c009ffc:	9732                	add	a4,a4,a2
1c009ffe:	0007a423          	sw	zero,8(a5)
1c00a002:	cbd8                	sw	a4,20(a5)
1c00a004:	8082                	ret
1c00a006:	0046a80b          	p.lw	a6,4(a3!)
1c00a00a:	1771                	addi	a4,a4,-4
1c00a00c:	0105a023          	sw	a6,0(a1)
1c00a010:	b7c1                	j	1c009fd0 <__rt_init_cluster_data+0x22>

1c00a012 <__rt_cluster_mount_step>:
1c00a012:	7179                	addi	sp,sp,-48
1c00a014:	ce4e                	sw	s3,28(sp)
1c00a016:	cc52                	sw	s4,24(sp)
1c00a018:	00800993          	li	s3,8
1c00a01c:	1c008a37          	lui	s4,0x1c008
1c00a020:	d422                	sw	s0,40(sp)
1c00a022:	ca56                	sw	s5,20(sp)
1c00a024:	d606                	sw	ra,44(sp)
1c00a026:	d226                	sw	s1,36(sp)
1c00a028:	d04a                	sw	s2,32(sp)
1c00a02a:	c85a                	sw	s6,16(sp)
1c00a02c:	842a                	mv	s0,a0
1c00a02e:	e6c9b9b3          	p.bclr	s3,s3,19,12
1c00a032:	1c001ab7          	lui	s5,0x1c001
1c00a036:	080a0a13          	addi	s4,s4,128 # 1c008080 <_start>
1c00a03a:	4c5c                	lw	a5,28(s0)
1c00a03c:	0217ad63          	p.beqimm	a5,1,1c00a076 <__rt_cluster_mount_step+0x64>
1c00a040:	0a27af63          	p.beqimm	a5,2,1c00a0fe <__rt_cluster_mount_step+0xec>
1c00a044:	ebcd                	bnez	a5,1c00a0f6 <__rt_cluster_mount_step+0xe4>
1c00a046:	5018                	lw	a4,32(s0)
1c00a048:	00042c23          	sw	zero,24(s0)
1c00a04c:	e719                	bnez	a4,1c00a05a <__rt_cluster_mount_step+0x48>
1c00a04e:	5048                	lw	a0,36(s0)
1c00a050:	006c                	addi	a1,sp,12
1c00a052:	c602                	sw	zero,12(sp)
1c00a054:	33c1                	jal	1c009e14 <__rt_pmu_cluster_power_up>
1c00a056:	47b2                	lw	a5,12(sp)
1c00a058:	cc08                	sw	a0,24(s0)
1c00a05a:	4c58                	lw	a4,28(s0)
1c00a05c:	0705                	addi	a4,a4,1
1c00a05e:	cc58                	sw	a4,28(s0)
1c00a060:	dfe9                	beqz	a5,1c00a03a <__rt_cluster_mount_step+0x28>
1c00a062:	50b2                	lw	ra,44(sp)
1c00a064:	5422                	lw	s0,40(sp)
1c00a066:	5492                	lw	s1,36(sp)
1c00a068:	5902                	lw	s2,32(sp)
1c00a06a:	49f2                	lw	s3,28(sp)
1c00a06c:	4a62                	lw	s4,24(sp)
1c00a06e:	4ad2                	lw	s5,20(sp)
1c00a070:	4b42                	lw	s6,16(sp)
1c00a072:	6145                	addi	sp,sp,48
1c00a074:	8082                	ret
1c00a076:	02042b03          	lw	s6,32(s0)
1c00a07a:	040b0493          	addi	s1,s6,64
1c00a07e:	04da                	slli	s1,s1,0x16
1c00a080:	009987b3          	add	a5,s3,s1
1c00a084:	0007a223          	sw	zero,4(a5)
1c00a088:	0007a423          	sw	zero,8(a5)
1c00a08c:	0007a023          	sw	zero,0(a5)
1c00a090:	644aa783          	lw	a5,1604(s5) # 1c001644 <__rt_platform>
1c00a094:	0017af63          	p.beqimm	a5,1,1c00a0b2 <__rt_cluster_mount_step+0xa0>
1c00a098:	4509                	li	a0,2
1c00a09a:	793000ef          	jal	ra,1c00b02c <__rt_fll_init>
1c00a09e:	1c0017b7          	lui	a5,0x1c001
1c00a0a2:	7e878793          	addi	a5,a5,2024 # 1c0017e8 <__rt_freq_domains>
1c00a0a6:	478c                	lw	a1,8(a5)
1c00a0a8:	c9a9                	beqz	a1,1c00a0fa <__rt_cluster_mount_step+0xe8>
1c00a0aa:	4601                	li	a2,0
1c00a0ac:	4509                	li	a0,2
1c00a0ae:	056010ef          	jal	ra,1c00b104 <rt_freq_set_and_get>
1c00a0b2:	00200937          	lui	s2,0x200
1c00a0b6:	01248733          	add	a4,s1,s2
1c00a0ba:	4785                	li	a5,1
1c00a0bc:	02f72023          	sw	a5,32(a4)
1c00a0c0:	855a                	mv	a0,s6
1c00a0c2:	35f5                	jal	1c009fae <__rt_init_cluster_data>
1c00a0c4:	855a                	mv	a0,s6
1c00a0c6:	927ff0ef          	jal	ra,1c0099ec <__rt_alloc_init_l1>
1c00a0ca:	002017b7          	lui	a5,0x201
1c00a0ce:	40078793          	addi	a5,a5,1024 # 201400 <__l1_heap_size+0x1e5418>
1c00a0d2:	577d                	li	a4,-1
1c00a0d4:	04090913          	addi	s2,s2,64 # 200040 <__l1_heap_size+0x1e4058>
1c00a0d8:	00e4e7a3          	p.sw	a4,a5(s1)
1c00a0dc:	9926                	add	s2,s2,s1
1c00a0de:	009250fb          	lp.setupi	x1,9,1c00a0e6 <__rt_cluster_mount_step+0xd4>
1c00a0e2:	0149222b          	p.sw	s4,4(s2!)
1c00a0e6:	0001                	nop
1c00a0e8:	002007b7          	lui	a5,0x200
1c00a0ec:	07a1                	addi	a5,a5,8
1c00a0ee:	1ff00713          	li	a4,511
1c00a0f2:	00e4e7a3          	p.sw	a4,a5(s1)
1c00a0f6:	4781                	li	a5,0
1c00a0f8:	b78d                	j	1c00a05a <__rt_cluster_mount_step+0x48>
1c00a0fa:	c788                	sw	a0,8(a5)
1c00a0fc:	bf5d                	j	1c00a0b2 <__rt_cluster_mount_step+0xa0>
1c00a0fe:	505c                	lw	a5,36(s0)
1c00a100:	5b98                	lw	a4,48(a5)
1c00a102:	d398                	sw	a4,32(a5)
1c00a104:	5798                	lw	a4,40(a5)
1c00a106:	c398                	sw	a4,0(a5)
1c00a108:	57d8                	lw	a4,44(a5)
1c00a10a:	c3d8                	sw	a4,4(a5)
1c00a10c:	0207a823          	sw	zero,48(a5) # 200030 <__l1_heap_size+0x1e4048>
1c00a110:	505c                	lw	a5,36(s0)
1c00a112:	00c02703          	lw	a4,12(zero) # c <__rt_sched>
1c00a116:	0007ac23          	sw	zero,24(a5)
1c00a11a:	cb01                	beqz	a4,1c00a12a <__rt_cluster_mount_step+0x118>
1c00a11c:	01002703          	lw	a4,16(zero) # 10 <__rt_sched+0x4>
1c00a120:	cf1c                	sw	a5,24(a4)
1c00a122:	00f02823          	sw	a5,16(zero) # 10 <__rt_sched+0x4>
1c00a126:	4785                	li	a5,1
1c00a128:	bf0d                	j	1c00a05a <__rt_cluster_mount_step+0x48>
1c00a12a:	00f02623          	sw	a5,12(zero) # c <__rt_sched>
1c00a12e:	bfd5                	j	1c00a122 <__rt_cluster_mount_step+0x110>

1c00a130 <__rt_cluster_init>:
1c00a130:	1c001537          	lui	a0,0x1c001
1c00a134:	1141                	addi	sp,sp,-16
1c00a136:	02800613          	li	a2,40
1c00a13a:	4581                	li	a1,0
1c00a13c:	7b450513          	addi	a0,a0,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c00a140:	c606                	sw	ra,12(sp)
1c00a142:	540010ef          	jal	ra,1c00b682 <memset>
1c00a146:	1c0095b7          	lui	a1,0x1c009
1c00a14a:	ca658593          	addi	a1,a1,-858 # 1c008ca6 <__rt_remote_enqueue_event>
1c00a14e:	4505                	li	a0,1
1c00a150:	003000ef          	jal	ra,1c00a952 <rt_irq_set_handler>
1c00a154:	477d                	li	a4,31
1c00a156:	f14027f3          	csrr	a5,mhartid
1c00a15a:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a15e:	02e79c63          	bne	a5,a4,1c00a196 <__rt_cluster_init+0x66>
1c00a162:	1a1097b7          	lui	a5,0x1a109
1c00a166:	4709                	li	a4,2
1c00a168:	c3d8                	sw	a4,4(a5)
1c00a16a:	1c0095b7          	lui	a1,0x1c009
1c00a16e:	c6e58593          	addi	a1,a1,-914 # 1c008c6e <__rt_bridge_enqueue_event>
1c00a172:	4511                	li	a0,4
1c00a174:	7de000ef          	jal	ra,1c00a952 <rt_irq_set_handler>
1c00a178:	477d                	li	a4,31
1c00a17a:	f14027f3          	csrr	a5,mhartid
1c00a17e:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a182:	00e79f63          	bne	a5,a4,1c00a1a0 <__rt_cluster_init+0x70>
1c00a186:	1a1097b7          	lui	a5,0x1a109
1c00a18a:	4741                	li	a4,16
1c00a18c:	c3d8                	sw	a4,4(a5)
1c00a18e:	40b2                	lw	ra,12(sp)
1c00a190:	4501                	li	a0,0
1c00a192:	0141                	addi	sp,sp,16
1c00a194:	8082                	ret
1c00a196:	002047b7          	lui	a5,0x204
1c00a19a:	4709                	li	a4,2
1c00a19c:	cbd8                	sw	a4,20(a5)
1c00a19e:	b7f1                	j	1c00a16a <__rt_cluster_init+0x3a>
1c00a1a0:	002047b7          	lui	a5,0x204
1c00a1a4:	4741                	li	a4,16
1c00a1a6:	cbd8                	sw	a4,20(a5)
1c00a1a8:	b7dd                	j	1c00a18e <__rt_cluster_init+0x5e>

1c00a1aa <pi_cluster_conf_init>:
1c00a1aa:	00052223          	sw	zero,4(a0)
1c00a1ae:	8082                	ret

1c00a1b0 <pi_cluster_open>:
1c00a1b0:	1101                	addi	sp,sp,-32
1c00a1b2:	ce06                	sw	ra,28(sp)
1c00a1b4:	cc22                	sw	s0,24(sp)
1c00a1b6:	ca26                	sw	s1,20(sp)
1c00a1b8:	c84a                	sw	s2,16(sp)
1c00a1ba:	c64e                	sw	s3,12(sp)
1c00a1bc:	30047973          	csrrci	s2,mstatus,8
1c00a1c0:	00452983          	lw	s3,4(a0)
1c00a1c4:	1c0014b7          	lui	s1,0x1c001
1c00a1c8:	02800793          	li	a5,40
1c00a1cc:	0049a703          	lw	a4,4(s3)
1c00a1d0:	7b448493          	addi	s1,s1,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c00a1d4:	42f704b3          	p.mac	s1,a4,a5
1c00a1d8:	c504                	sw	s1,8(a0)
1c00a1da:	bf6ff0ef          	jal	ra,1c0095d0 <__rt_wait_event_prepare_blocking>
1c00a1de:	477d                	li	a4,31
1c00a1e0:	f14027f3          	csrr	a5,mhartid
1c00a1e4:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a1e8:	842a                	mv	s0,a0
1c00a1ea:	04e79463          	bne	a5,a4,1c00a232 <pi_cluster_open+0x82>
1c00a1ee:	511c                	lw	a5,32(a0)
1c00a1f0:	0004ae23          	sw	zero,28(s1)
1c00a1f4:	d0c8                	sw	a0,36(s1)
1c00a1f6:	d91c                	sw	a5,48(a0)
1c00a1f8:	411c                	lw	a5,0(a0)
1c00a1fa:	02052223          	sw	zero,36(a0)
1c00a1fe:	d51c                	sw	a5,40(a0)
1c00a200:	415c                	lw	a5,4(a0)
1c00a202:	c144                	sw	s1,4(a0)
1c00a204:	d55c                	sw	a5,44(a0)
1c00a206:	1c00a7b7          	lui	a5,0x1c00a
1c00a20a:	01278793          	addi	a5,a5,18 # 1c00a012 <__rt_cluster_mount_step>
1c00a20e:	c11c                	sw	a5,0(a0)
1c00a210:	4785                	li	a5,1
1c00a212:	d11c                	sw	a5,32(a0)
1c00a214:	8526                	mv	a0,s1
1c00a216:	3bf5                	jal	1c00a012 <__rt_cluster_mount_step>
1c00a218:	8522                	mv	a0,s0
1c00a21a:	d04ff0ef          	jal	ra,1c00971e <__rt_wait_event>
1c00a21e:	30091073          	csrw	mstatus,s2
1c00a222:	40f2                	lw	ra,28(sp)
1c00a224:	4462                	lw	s0,24(sp)
1c00a226:	44d2                	lw	s1,20(sp)
1c00a228:	4942                	lw	s2,16(sp)
1c00a22a:	49b2                	lw	s3,12(sp)
1c00a22c:	4501                	li	a0,0
1c00a22e:	6105                	addi	sp,sp,32
1c00a230:	8082                	ret
1c00a232:	0049a483          	lw	s1,4(s3)
1c00a236:	8526                	mv	a0,s1
1c00a238:	3b9d                	jal	1c009fae <__rt_init_cluster_data>
1c00a23a:	04048513          	addi	a0,s1,64
1c00a23e:	002017b7          	lui	a5,0x201
1c00a242:	055a                	slli	a0,a0,0x16
1c00a244:	40078793          	addi	a5,a5,1024 # 201400 <__l1_heap_size+0x1e5418>
1c00a248:	577d                	li	a4,-1
1c00a24a:	00e567a3          	p.sw	a4,a5(a0)
1c00a24e:	002007b7          	lui	a5,0x200
1c00a252:	04478793          	addi	a5,a5,68 # 200044 <__l1_heap_size+0x1e405c>
1c00a256:	1c0086b7          	lui	a3,0x1c008
1c00a25a:	97aa                	add	a5,a5,a0
1c00a25c:	08068693          	addi	a3,a3,128 # 1c008080 <_start>
1c00a260:	008250fb          	lp.setupi	x1,8,1c00a268 <pi_cluster_open+0xb8>
1c00a264:	00d7a22b          	p.sw	a3,4(a5!)
1c00a268:	0001                	nop
1c00a26a:	002007b7          	lui	a5,0x200
1c00a26e:	07a1                	addi	a5,a5,8
1c00a270:	577d                	li	a4,-1
1c00a272:	00e567a3          	p.sw	a4,a5(a0)
1c00a276:	8522                	mv	a0,s0
1c00a278:	c22ff0ef          	jal	ra,1c00969a <rt_event_push>
1c00a27c:	bf71                	j	1c00a218 <pi_cluster_open+0x68>

1c00a27e <pi_cluster_close>:
1c00a27e:	451c                	lw	a5,8(a0)
1c00a280:	1101                	addi	sp,sp,-32
1c00a282:	cc22                	sw	s0,24(sp)
1c00a284:	5380                	lw	s0,32(a5)
1c00a286:	1c0017b7          	lui	a5,0x1c001
1c00a28a:	6447a783          	lw	a5,1604(a5) # 1c001644 <__rt_platform>
1c00a28e:	ce06                	sw	ra,28(sp)
1c00a290:	0017a563          	p.beqimm	a5,1,1c00a29a <pi_cluster_close+0x1c>
1c00a294:	4509                	li	a0,2
1c00a296:	63b000ef          	jal	ra,1c00b0d0 <__rt_fll_deinit>
1c00a29a:	c602                	sw	zero,12(sp)
1c00a29c:	e401                	bnez	s0,1c00a2a4 <pi_cluster_close+0x26>
1c00a29e:	006c                	addi	a1,sp,12
1c00a2a0:	4501                	li	a0,0
1c00a2a2:	36ad                	jal	1c009e0c <__rt_pmu_cluster_power_down>
1c00a2a4:	40f2                	lw	ra,28(sp)
1c00a2a6:	4462                	lw	s0,24(sp)
1c00a2a8:	4501                	li	a0,0
1c00a2aa:	6105                	addi	sp,sp,32
1c00a2ac:	8082                	ret

1c00a2ae <__rt_cluster_push_fc_event>:
1c00a2ae:	002047b7          	lui	a5,0x204
1c00a2b2:	0c078793          	addi	a5,a5,192 # 2040c0 <__l1_heap_size+0x1e80d8>
1c00a2b6:	0007e703          	p.elw	a4,0(a5)
1c00a2ba:	f1402773          	csrr	a4,mhartid
1c00a2be:	1c0017b7          	lui	a5,0x1c001
1c00a2c2:	8715                	srai	a4,a4,0x5
1c00a2c4:	f2673733          	p.bclr	a4,a4,25,6
1c00a2c8:	02800693          	li	a3,40
1c00a2cc:	7b478793          	addi	a5,a5,1972 # 1c0017b4 <__rt_fc_cluster_data>
1c00a2d0:	42d707b3          	p.mac	a5,a4,a3
1c00a2d4:	4689                	li	a3,2
1c00a2d6:	00204737          	lui	a4,0x204
1c00a2da:	43d0                	lw	a2,4(a5)
1c00a2dc:	ea19                	bnez	a2,1c00a2f2 <__rt_cluster_push_fc_event+0x44>
1c00a2de:	c3c8                	sw	a0,4(a5)
1c00a2e0:	4709                	li	a4,2
1c00a2e2:	1a1097b7          	lui	a5,0x1a109
1c00a2e6:	cb98                	sw	a4,16(a5)
1c00a2e8:	002047b7          	lui	a5,0x204
1c00a2ec:	0c07a023          	sw	zero,192(a5) # 2040c0 <__l1_heap_size+0x1e80d8>
1c00a2f0:	8082                	ret
1c00a2f2:	c714                	sw	a3,8(a4)
1c00a2f4:	03c76603          	p.elw	a2,60(a4) # 20403c <__l1_heap_size+0x1e8054>
1c00a2f8:	c354                	sw	a3,4(a4)
1c00a2fa:	b7c5                	j	1c00a2da <__rt_cluster_push_fc_event+0x2c>

1c00a2fc <__rt_cluster_new>:
1c00a2fc:	1c00a5b7          	lui	a1,0x1c00a
1c00a300:	1141                	addi	sp,sp,-16
1c00a302:	4601                	li	a2,0
1c00a304:	13058593          	addi	a1,a1,304 # 1c00a130 <__rt_cluster_init>
1c00a308:	4501                	li	a0,0
1c00a30a:	c606                	sw	ra,12(sp)
1c00a30c:	7d4000ef          	jal	ra,1c00aae0 <__rt_cbsys_add>
1c00a310:	c10d                	beqz	a0,1c00a332 <__rt_cluster_new+0x36>
1c00a312:	f1402673          	csrr	a2,mhartid
1c00a316:	1c001537          	lui	a0,0x1c001
1c00a31a:	40565593          	srai	a1,a2,0x5
1c00a31e:	f265b5b3          	p.bclr	a1,a1,25,6
1c00a322:	f4563633          	p.bclr	a2,a2,26,5
1c00a326:	a4050513          	addi	a0,a0,-1472 # 1c000a40 <PIo2+0xe4>
1c00a32a:	5b2010ef          	jal	ra,1c00b8dc <printf>
1c00a32e:	53c010ef          	jal	ra,1c00b86a <abort>
1c00a332:	40b2                	lw	ra,12(sp)
1c00a334:	0141                	addi	sp,sp,16
1c00a336:	8082                	ret

1c00a338 <__rt_cluster_pulpos_emu_init>:
1c00a338:	1141                	addi	sp,sp,-16
1c00a33a:	45b1                	li	a1,12
1c00a33c:	4501                	li	a0,0
1c00a33e:	c606                	sw	ra,12(sp)
1c00a340:	e3eff0ef          	jal	ra,1c00997e <rt_alloc>
1c00a344:	1c0017b7          	lui	a5,0x1c001
1c00a348:	74a7a023          	sw	a0,1856(a5) # 1c001740 <__rt_fc_cluster_device>
1c00a34c:	e10d                	bnez	a0,1c00a36e <__rt_cluster_pulpos_emu_init+0x36>
1c00a34e:	f1402673          	csrr	a2,mhartid
1c00a352:	1c001537          	lui	a0,0x1c001
1c00a356:	40565593          	srai	a1,a2,0x5
1c00a35a:	f265b5b3          	p.bclr	a1,a1,25,6
1c00a35e:	f4563633          	p.bclr	a2,a2,26,5
1c00a362:	a8850513          	addi	a0,a0,-1400 # 1c000a88 <PIo2+0x12c>
1c00a366:	576010ef          	jal	ra,1c00b8dc <printf>
1c00a36a:	500010ef          	jal	ra,1c00b86a <abort>
1c00a36e:	40b2                	lw	ra,12(sp)
1c00a370:	0141                	addi	sp,sp,16
1c00a372:	8082                	ret

1c00a374 <rt_cluster_call>:
1c00a374:	7139                	addi	sp,sp,-64
1c00a376:	d84a                	sw	s2,48(sp)
1c00a378:	4906                	lw	s2,64(sp)
1c00a37a:	dc22                	sw	s0,56(sp)
1c00a37c:	842e                	mv	s0,a1
1c00a37e:	de06                	sw	ra,60(sp)
1c00a380:	da26                	sw	s1,52(sp)
1c00a382:	d64e                	sw	s3,44(sp)
1c00a384:	300479f3          	csrrci	s3,mstatus,8
1c00a388:	84ca                	mv	s1,s2
1c00a38a:	02091163          	bnez	s2,1c00a3ac <rt_cluster_call+0x38>
1c00a38e:	ce32                	sw	a2,28(sp)
1c00a390:	cc36                	sw	a3,24(sp)
1c00a392:	ca3a                	sw	a4,20(sp)
1c00a394:	c83e                	sw	a5,16(sp)
1c00a396:	c642                	sw	a6,12(sp)
1c00a398:	c446                	sw	a7,8(sp)
1c00a39a:	a36ff0ef          	jal	ra,1c0095d0 <__rt_wait_event_prepare_blocking>
1c00a39e:	48a2                	lw	a7,8(sp)
1c00a3a0:	4832                	lw	a6,12(sp)
1c00a3a2:	47c2                	lw	a5,16(sp)
1c00a3a4:	4752                	lw	a4,20(sp)
1c00a3a6:	46e2                	lw	a3,24(sp)
1c00a3a8:	4672                	lw	a2,28(sp)
1c00a3aa:	84aa                	mv	s1,a0
1c00a3ac:	1c0015b7          	lui	a1,0x1c001
1c00a3b0:	65058513          	addi	a0,a1,1616 # 1c001650 <_edata>
1c00a3b4:	c55c                	sw	a5,12(a0)
1c00a3b6:	1c0017b7          	lui	a5,0x1c001
1c00a3ba:	c110                	sw	a2,0(a0)
1c00a3bc:	c154                	sw	a3,4(a0)
1c00a3be:	c518                	sw	a4,8(a0)
1c00a3c0:	01052823          	sw	a6,16(a0)
1c00a3c4:	01152a23          	sw	a7,20(a0)
1c00a3c8:	7407a503          	lw	a0,1856(a5) # 1c001740 <__rt_fc_cluster_device>
1c00a3cc:	47b1                	li	a5,12
1c00a3ce:	8626                	mv	a2,s1
1c00a3d0:	42f40533          	p.mac	a0,s0,a5
1c00a3d4:	65058593          	addi	a1,a1,1616
1c00a3d8:	2041                	jal	1c00a458 <pi_cluster_send_task_to_cl_async>
1c00a3da:	842a                	mv	s0,a0
1c00a3dc:	cd01                	beqz	a0,1c00a3f4 <rt_cluster_call+0x80>
1c00a3de:	30099073          	csrw	mstatus,s3
1c00a3e2:	547d                	li	s0,-1
1c00a3e4:	8522                	mv	a0,s0
1c00a3e6:	50f2                	lw	ra,60(sp)
1c00a3e8:	5462                	lw	s0,56(sp)
1c00a3ea:	54d2                	lw	s1,52(sp)
1c00a3ec:	5942                	lw	s2,48(sp)
1c00a3ee:	59b2                	lw	s3,44(sp)
1c00a3f0:	6121                	addi	sp,sp,64
1c00a3f2:	8082                	ret
1c00a3f4:	00091563          	bnez	s2,1c00a3fe <rt_cluster_call+0x8a>
1c00a3f8:	8526                	mv	a0,s1
1c00a3fa:	b24ff0ef          	jal	ra,1c00971e <__rt_wait_event>
1c00a3fe:	30099073          	csrw	mstatus,s3
1c00a402:	b7cd                	j	1c00a3e4 <rt_cluster_call+0x70>

1c00a404 <rt_cluster_mount>:
1c00a404:	7139                	addi	sp,sp,-64
1c00a406:	dc22                	sw	s0,56(sp)
1c00a408:	da26                	sw	s1,52(sp)
1c00a40a:	d84a                	sw	s2,48(sp)
1c00a40c:	4431                	li	s0,12
1c00a40e:	1c0014b7          	lui	s1,0x1c001
1c00a412:	de06                	sw	ra,60(sp)
1c00a414:	d64e                	sw	s3,44(sp)
1c00a416:	8936                	mv	s2,a3
1c00a418:	02858433          	mul	s0,a1,s0
1c00a41c:	74048493          	addi	s1,s1,1856 # 1c001740 <__rt_fc_cluster_device>
1c00a420:	c905                	beqz	a0,1c00a450 <rt_cluster_mount+0x4c>
1c00a422:	0068                	addi	a0,sp,12
1c00a424:	89ae                	mv	s3,a1
1c00a426:	3351                	jal	1c00a1aa <pi_cluster_conf_init>
1c00a428:	4088                	lw	a0,0(s1)
1c00a42a:	006c                	addi	a1,sp,12
1c00a42c:	9522                	add	a0,a0,s0
1c00a42e:	2305                	jal	1c00a94e <pi_open_from_conf>
1c00a430:	4088                	lw	a0,0(s1)
1c00a432:	c84e                	sw	s3,16(sp)
1c00a434:	9522                	add	a0,a0,s0
1c00a436:	3bad                	jal	1c00a1b0 <pi_cluster_open>
1c00a438:	00090563          	beqz	s2,1c00a442 <rt_cluster_mount+0x3e>
1c00a43c:	854a                	mv	a0,s2
1c00a43e:	a5cff0ef          	jal	ra,1c00969a <rt_event_push>
1c00a442:	50f2                	lw	ra,60(sp)
1c00a444:	5462                	lw	s0,56(sp)
1c00a446:	54d2                	lw	s1,52(sp)
1c00a448:	5942                	lw	s2,48(sp)
1c00a44a:	59b2                	lw	s3,44(sp)
1c00a44c:	6121                	addi	sp,sp,64
1c00a44e:	8082                	ret
1c00a450:	4088                	lw	a0,0(s1)
1c00a452:	9522                	add	a0,a0,s0
1c00a454:	352d                	jal	1c00a27e <pi_cluster_close>
1c00a456:	b7cd                	j	1c00a438 <rt_cluster_mount+0x34>

1c00a458 <pi_cluster_send_task_to_cl_async>:
1c00a458:	1101                	addi	sp,sp,-32
1c00a45a:	ca26                	sw	s1,20(sp)
1c00a45c:	4504                	lw	s1,8(a0)
1c00a45e:	cc22                	sw	s0,24(sp)
1c00a460:	c256                	sw	s5,4(sp)
1c00a462:	842e                	mv	s0,a1
1c00a464:	8ab2                	mv	s5,a2
1c00a466:	ce06                	sw	ra,28(sp)
1c00a468:	c84a                	sw	s2,16(sp)
1c00a46a:	c64e                	sw	s3,12(sp)
1c00a46c:	c452                	sw	s4,8(sp)
1c00a46e:	30047a73          	csrrci	s4,mstatus,8
1c00a472:	00060823          	sb	zero,16(a2) # 10010 <_l1_preload_size+0xc000>
1c00a476:	4785                	li	a5,1
1c00a478:	d1dc                	sw	a5,36(a1)
1c00a47a:	49dc                	lw	a5,20(a1)
1c00a47c:	0144a983          	lw	s3,20(s1)
1c00a480:	e399                	bnez	a5,1c00a486 <pi_cluster_send_task_to_cl_async+0x2e>
1c00a482:	47a5                	li	a5,9
1c00a484:	c9dc                	sw	a5,20(a1)
1c00a486:	441c                	lw	a5,8(s0)
1c00a488:	ef85                	bnez	a5,1c00a4c0 <pi_cluster_send_task_to_cl_async+0x68>
1c00a48a:	445c                	lw	a5,12(s0)
1c00a48c:	eb81                	bnez	a5,1c00a49c <pi_cluster_send_task_to_cl_async+0x44>
1c00a48e:	6785                	lui	a5,0x1
1c00a490:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c00a494:	c45c                	sw	a5,12(s0)
1c00a496:	40000793          	li	a5,1024
1c00a49a:	c81c                	sw	a5,16(s0)
1c00a49c:	481c                	lw	a5,16(s0)
1c00a49e:	00c42903          	lw	s2,12(s0)
1c00a4a2:	e399                	bnez	a5,1c00a4a8 <pi_cluster_send_task_to_cl_async+0x50>
1c00a4a4:	01242823          	sw	s2,16(s0)
1c00a4a8:	485c                	lw	a5,20(s0)
1c00a4aa:	4818                	lw	a4,16(s0)
1c00a4ac:	448c                	lw	a1,8(s1)
1c00a4ae:	17fd                	addi	a5,a5,-1
1c00a4b0:	42e78933          	p.mac	s2,a5,a4
1c00a4b4:	cdad                	beqz	a1,1c00a52e <pi_cluster_send_task_to_cl_async+0xd6>
1c00a4b6:	44d0                	lw	a2,12(s1)
1c00a4b8:	07261163          	bne	a2,s2,1c00a51a <pi_cluster_send_task_to_cl_async+0xc2>
1c00a4bc:	449c                	lw	a5,8(s1)
1c00a4be:	c41c                	sw	a5,8(s0)
1c00a4c0:	485c                	lw	a5,20(s0)
1c00a4c2:	01542c23          	sw	s5,24(s0)
1c00a4c6:	02042023          	sw	zero,32(s0)
1c00a4ca:	fff78713          	addi	a4,a5,-1
1c00a4ce:	4785                	li	a5,1
1c00a4d0:	00e797b3          	sll	a5,a5,a4
1c00a4d4:	17fd                	addi	a5,a5,-1
1c00a4d6:	d41c                	sw	a5,40(s0)
1c00a4d8:	0089a783          	lw	a5,8(s3)
1c00a4dc:	cfa5                	beqz	a5,1c00a554 <pi_cluster_send_task_to_cl_async+0xfc>
1c00a4de:	d380                	sw	s0,32(a5)
1c00a4e0:	0089a423          	sw	s0,8(s3)
1c00a4e4:	0009a783          	lw	a5,0(s3)
1c00a4e8:	e399                	bnez	a5,1c00a4ee <pi_cluster_send_task_to_cl_async+0x96>
1c00a4ea:	0089a023          	sw	s0,0(s3)
1c00a4ee:	509c                	lw	a5,32(s1)
1c00a4f0:	00201737          	lui	a4,0x201
1c00a4f4:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e4e1c>
1c00a4f8:	04078793          	addi	a5,a5,64
1c00a4fc:	07da                	slli	a5,a5,0x16
1c00a4fe:	0007e723          	p.sw	zero,a4(a5)
1c00a502:	300a1073          	csrw	mstatus,s4
1c00a506:	4501                	li	a0,0
1c00a508:	40f2                	lw	ra,28(sp)
1c00a50a:	4462                	lw	s0,24(sp)
1c00a50c:	44d2                	lw	s1,20(sp)
1c00a50e:	4942                	lw	s2,16(sp)
1c00a510:	49b2                	lw	s3,12(sp)
1c00a512:	4a22                	lw	s4,8(sp)
1c00a514:	4a92                	lw	s5,4(sp)
1c00a516:	6105                	addi	sp,sp,32
1c00a518:	8082                	ret
1c00a51a:	1c001737          	lui	a4,0x1c001
1c00a51e:	76072503          	lw	a0,1888(a4) # 1c001760 <__rt_alloc_l1>
1c00a522:	509c                	lw	a5,32(s1)
1c00a524:	4761                	li	a4,24
1c00a526:	42f70533          	p.mac	a0,a4,a5
1c00a52a:	bcaff0ef          	jal	ra,1c0098f4 <rt_user_free>
1c00a52e:	1c001737          	lui	a4,0x1c001
1c00a532:	76072503          	lw	a0,1888(a4) # 1c001760 <__rt_alloc_l1>
1c00a536:	509c                	lw	a5,32(s1)
1c00a538:	4761                	li	a4,24
1c00a53a:	0124a623          	sw	s2,12(s1)
1c00a53e:	42f70533          	p.mac	a0,a4,a5
1c00a542:	85ca                	mv	a1,s2
1c00a544:	b3eff0ef          	jal	ra,1c009882 <rt_user_alloc>
1c00a548:	c488                	sw	a0,8(s1)
1c00a54a:	f92d                	bnez	a0,1c00a4bc <pi_cluster_send_task_to_cl_async+0x64>
1c00a54c:	300a1073          	csrw	mstatus,s4
1c00a550:	557d                	li	a0,-1
1c00a552:	bf5d                	j	1c00a508 <pi_cluster_send_task_to_cl_async+0xb0>
1c00a554:	0089a223          	sw	s0,4(s3)
1c00a558:	b761                	j	1c00a4e0 <pi_cluster_send_task_to_cl_async+0x88>

1c00a55a <cpu_perf_get>:
1c00a55a:	10e52763          	p.beqimm	a0,14,1c00a668 <cpu_perf_get+0x10e>
1c00a55e:	47b9                	li	a5,14
1c00a560:	04a7ee63          	bltu	a5,a0,1c00a5bc <cpu_perf_get+0x62>
1c00a564:	0e652063          	p.beqimm	a0,6,1c00a644 <cpu_perf_get+0xea>
1c00a568:	4799                	li	a5,6
1c00a56a:	02a7e463          	bltu	a5,a0,1c00a592 <cpu_perf_get+0x38>
1c00a56e:	0c252263          	p.beqimm	a0,2,1c00a632 <cpu_perf_get+0xd8>
1c00a572:	4789                	li	a5,2
1c00a574:	00a7e763          	bltu	a5,a0,1c00a582 <cpu_perf_get+0x28>
1c00a578:	c55d                	beqz	a0,1c00a626 <cpu_perf_get+0xcc>
1c00a57a:	0a152963          	p.beqimm	a0,1,1c00a62c <cpu_perf_get+0xd2>
1c00a57e:	4501                	li	a0,0
1c00a580:	8082                	ret
1c00a582:	0a452e63          	p.beqimm	a0,4,1c00a63e <cpu_perf_get+0xe4>
1c00a586:	4791                	li	a5,4
1c00a588:	0aa7f863          	bleu	a0,a5,1c00a638 <cpu_perf_get+0xde>
1c00a58c:	78502573          	csrr	a0,pccr5
1c00a590:	8082                	ret
1c00a592:	0ca52263          	p.beqimm	a0,10,1c00a656 <cpu_perf_get+0xfc>
1c00a596:	47a9                	li	a5,10
1c00a598:	00a7ea63          	bltu	a5,a0,1c00a5ac <cpu_perf_get+0x52>
1c00a59c:	0a852a63          	p.beqimm	a0,8,1c00a650 <cpu_perf_get+0xf6>
1c00a5a0:	47a1                	li	a5,8
1c00a5a2:	0aa7f463          	bleu	a0,a5,1c00a64a <cpu_perf_get+0xf0>
1c00a5a6:	78902573          	csrr	a0,pccr9
1c00a5aa:	8082                	ret
1c00a5ac:	0ac52b63          	p.beqimm	a0,12,1c00a662 <cpu_perf_get+0x108>
1c00a5b0:	47b1                	li	a5,12
1c00a5b2:	0aa7f563          	bleu	a0,a5,1c00a65c <cpu_perf_get+0x102>
1c00a5b6:	78d02573          	csrr	a0,pccr13
1c00a5ba:	8082                	ret
1c00a5bc:	47dd                	li	a5,23
1c00a5be:	0cf50763          	beq	a0,a5,1c00a68c <cpu_perf_get+0x132>
1c00a5c2:	02a7ea63          	bltu	a5,a0,1c00a5f6 <cpu_perf_get+0x9c>
1c00a5c6:	47cd                	li	a5,19
1c00a5c8:	0af50963          	beq	a0,a5,1c00a67a <cpu_perf_get+0x120>
1c00a5cc:	00a7ed63          	bltu	a5,a0,1c00a5e6 <cpu_perf_get+0x8c>
1c00a5d0:	47c1                	li	a5,16
1c00a5d2:	0af50163          	beq	a0,a5,1c00a674 <cpu_perf_get+0x11a>
1c00a5d6:	08f56c63          	bltu	a0,a5,1c00a66e <cpu_perf_get+0x114>
1c00a5da:	47c9                	li	a5,18
1c00a5dc:	faf511e3          	bne	a0,a5,1c00a57e <cpu_perf_get+0x24>
1c00a5e0:	79202573          	csrr	a0,pccr18
1c00a5e4:	8082                	ret
1c00a5e6:	47d5                	li	a5,21
1c00a5e8:	08f50f63          	beq	a0,a5,1c00a686 <cpu_perf_get+0x12c>
1c00a5ec:	08a7fa63          	bleu	a0,a5,1c00a680 <cpu_perf_get+0x126>
1c00a5f0:	79602573          	csrr	a0,pccr22
1c00a5f4:	8082                	ret
1c00a5f6:	47ed                	li	a5,27
1c00a5f8:	0af50363          	beq	a0,a5,1c00a69e <cpu_perf_get+0x144>
1c00a5fc:	00a7ea63          	bltu	a5,a0,1c00a610 <cpu_perf_get+0xb6>
1c00a600:	47e5                	li	a5,25
1c00a602:	08f50b63          	beq	a0,a5,1c00a698 <cpu_perf_get+0x13e>
1c00a606:	08a7f663          	bleu	a0,a5,1c00a692 <cpu_perf_get+0x138>
1c00a60a:	79a02573          	csrr	a0,pccr26
1c00a60e:	8082                	ret
1c00a610:	47f5                	li	a5,29
1c00a612:	08f50c63          	beq	a0,a5,1c00a6aa <cpu_perf_get+0x150>
1c00a616:	08f56763          	bltu	a0,a5,1c00a6a4 <cpu_perf_get+0x14a>
1c00a61a:	47f9                	li	a5,30
1c00a61c:	f6f511e3          	bne	a0,a5,1c00a57e <cpu_perf_get+0x24>
1c00a620:	79e02573          	csrr	a0,pccr30
1c00a624:	8082                	ret
1c00a626:	78002573          	csrr	a0,pccr0
1c00a62a:	8082                	ret
1c00a62c:	78102573          	csrr	a0,pccr1
1c00a630:	8082                	ret
1c00a632:	78202573          	csrr	a0,pccr2
1c00a636:	8082                	ret
1c00a638:	78302573          	csrr	a0,pccr3
1c00a63c:	8082                	ret
1c00a63e:	78402573          	csrr	a0,pccr4
1c00a642:	8082                	ret
1c00a644:	78602573          	csrr	a0,pccr6
1c00a648:	8082                	ret
1c00a64a:	78702573          	csrr	a0,pccr7
1c00a64e:	8082                	ret
1c00a650:	78802573          	csrr	a0,pccr8
1c00a654:	8082                	ret
1c00a656:	78a02573          	csrr	a0,pccr10
1c00a65a:	8082                	ret
1c00a65c:	78b02573          	csrr	a0,pccr11
1c00a660:	8082                	ret
1c00a662:	78c02573          	csrr	a0,pccr12
1c00a666:	8082                	ret
1c00a668:	78e02573          	csrr	a0,pccr14
1c00a66c:	8082                	ret
1c00a66e:	78f02573          	csrr	a0,pccr15
1c00a672:	8082                	ret
1c00a674:	79002573          	csrr	a0,pccr16
1c00a678:	8082                	ret
1c00a67a:	79302573          	csrr	a0,pccr19
1c00a67e:	8082                	ret
1c00a680:	79402573          	csrr	a0,pccr20
1c00a684:	8082                	ret
1c00a686:	79502573          	csrr	a0,pccr21
1c00a68a:	8082                	ret
1c00a68c:	79702573          	csrr	a0,pccr23
1c00a690:	8082                	ret
1c00a692:	79802573          	csrr	a0,pccr24
1c00a696:	8082                	ret
1c00a698:	79902573          	csrr	a0,pccr25
1c00a69c:	8082                	ret
1c00a69e:	79b02573          	csrr	a0,pccr27
1c00a6a2:	8082                	ret
1c00a6a4:	79c02573          	csrr	a0,pccr28
1c00a6a8:	8082                	ret
1c00a6aa:	79d02573          	csrr	a0,pccr29
1c00a6ae:	8082                	ret

1c00a6b0 <rt_perf_init>:
1c00a6b0:	0511                	addi	a0,a0,4
1c00a6b2:	012250fb          	lp.setupi	x1,18,1c00a6ba <rt_perf_init+0xa>
1c00a6b6:	0005222b          	p.sw	zero,4(a0!)
1c00a6ba:	0001                	nop
1c00a6bc:	8082                	ret

1c00a6be <rt_perf_conf>:
1c00a6be:	c10c                	sw	a1,0(a0)
1c00a6c0:	cc059073          	csrw	0xcc0,a1
1c00a6c4:	8082                	ret

1c00a6c6 <rt_perf_save>:
1c00a6c6:	7179                	addi	sp,sp,-48
1c00a6c8:	d04a                	sw	s2,32(sp)
1c00a6ca:	00052903          	lw	s2,0(a0)
1c00a6ce:	d226                	sw	s1,36(sp)
1c00a6d0:	ce4e                	sw	s3,28(sp)
1c00a6d2:	f14024f3          	csrr	s1,mhartid
1c00a6d6:	102009b7          	lui	s3,0x10200
1c00a6da:	8495                	srai	s1,s1,0x5
1c00a6dc:	cc52                	sw	s4,24(sp)
1c00a6de:	ca56                	sw	s5,20(sp)
1c00a6e0:	c85a                	sw	s6,16(sp)
1c00a6e2:	c65e                	sw	s7,12(sp)
1c00a6e4:	d606                	sw	ra,44(sp)
1c00a6e6:	d422                	sw	s0,40(sp)
1c00a6e8:	8baa                	mv	s7,a0
1c00a6ea:	4a85                	li	s5,1
1c00a6ec:	f264b4b3          	p.bclr	s1,s1,25,6
1c00a6f0:	4b7d                	li	s6,31
1c00a6f2:	4a45                	li	s4,17
1c00a6f4:	40098993          	addi	s3,s3,1024 # 10200400 <__l1_end+0x1fc3e8>
1c00a6f8:	00091d63          	bnez	s2,1c00a712 <rt_perf_save+0x4c>
1c00a6fc:	50b2                	lw	ra,44(sp)
1c00a6fe:	5422                	lw	s0,40(sp)
1c00a700:	5492                	lw	s1,36(sp)
1c00a702:	5902                	lw	s2,32(sp)
1c00a704:	49f2                	lw	s3,28(sp)
1c00a706:	4a62                	lw	s4,24(sp)
1c00a708:	4ad2                	lw	s5,20(sp)
1c00a70a:	4b42                	lw	s6,16(sp)
1c00a70c:	4bb2                	lw	s7,12(sp)
1c00a70e:	6145                	addi	sp,sp,48
1c00a710:	8082                	ret
1c00a712:	10091533          	p.fl1	a0,s2
1c00a716:	00aa97b3          	sll	a5,s5,a0
1c00a71a:	fff7c793          	not	a5,a5
1c00a71e:	00f97933          	and	s2,s2,a5
1c00a722:	00251413          	slli	s0,a0,0x2
1c00a726:	01649d63          	bne	s1,s6,1c00a740 <rt_perf_save+0x7a>
1c00a72a:	03451063          	bne	a0,s4,1c00a74a <rt_perf_save+0x84>
1c00a72e:	1a10b537          	lui	a0,0x1a10b
1c00a732:	00852503          	lw	a0,8(a0) # 1a10b008 <__l1_end+0xa106ff0>
1c00a736:	945e                	add	s0,s0,s7
1c00a738:	405c                	lw	a5,4(s0)
1c00a73a:	953e                	add	a0,a0,a5
1c00a73c:	c048                	sw	a0,4(s0)
1c00a73e:	bf6d                	j	1c00a6f8 <rt_perf_save+0x32>
1c00a740:	01451563          	bne	a0,s4,1c00a74a <rt_perf_save+0x84>
1c00a744:	0089a503          	lw	a0,8(s3)
1c00a748:	b7fd                	j	1c00a736 <rt_perf_save+0x70>
1c00a74a:	3d01                	jal	1c00a55a <cpu_perf_get>
1c00a74c:	b7ed                	j	1c00a736 <rt_perf_save+0x70>

1c00a74e <cluster_start>:
1c00a74e:	002047b7          	lui	a5,0x204
1c00a752:	00070737          	lui	a4,0x70
1c00a756:	c798                	sw	a4,8(a5)
1c00a758:	1ff00713          	li	a4,511
1c00a75c:	002046b7          	lui	a3,0x204
1c00a760:	08e6a223          	sw	a4,132(a3) # 204084 <__l1_heap_size+0x1e809c>
1c00a764:	20078693          	addi	a3,a5,512 # 204200 <__l1_heap_size+0x1e8218>
1c00a768:	c298                	sw	a4,0(a3)
1c00a76a:	20c78793          	addi	a5,a5,524
1c00a76e:	c398                	sw	a4,0(a5)
1c00a770:	8082                	ret

1c00a772 <__rt_init>:
1c00a772:	1101                	addi	sp,sp,-32
1c00a774:	ce06                	sw	ra,28(sp)
1c00a776:	cc22                	sw	s0,24(sp)
1c00a778:	2d19                	jal	1c00ad8e <__rt_bridge_set_available>
1c00a77a:	1c0017b7          	lui	a5,0x1c001
1c00a77e:	6447a783          	lw	a5,1604(a5) # 1c001644 <__rt_platform>
1c00a782:	0237b263          	p.bneimm	a5,3,1c00a7a6 <__rt_init+0x34>
1c00a786:	7d005073          	csrwi	0x7d0,0
1c00a78a:	1c0017b7          	lui	a5,0x1c001
1c00a78e:	c6078793          	addi	a5,a5,-928 # 1c000c60 <.got>
1c00a792:	7d179073          	csrw	0x7d1,a5
1c00a796:	1c0017b7          	lui	a5,0x1c001
1c00a79a:	46078793          	addi	a5,a5,1120 # 1c001460 <stack>
1c00a79e:	7d279073          	csrw	0x7d2,a5
1c00a7a2:	7d00d073          	csrwi	0x7d0,1
1c00a7a6:	24b1                	jal	1c00a9f2 <__rt_irq_init>
1c00a7a8:	1a1067b7          	lui	a5,0x1a106
1c00a7ac:	577d                	li	a4,-1
1c00a7ae:	00478693          	addi	a3,a5,4 # 1a106004 <__l1_end+0xa101fec>
1c00a7b2:	c298                	sw	a4,0(a3)
1c00a7b4:	00878693          	addi	a3,a5,8
1c00a7b8:	c298                	sw	a4,0(a3)
1c00a7ba:	00c78693          	addi	a3,a5,12
1c00a7be:	c298                	sw	a4,0(a3)
1c00a7c0:	01078693          	addi	a3,a5,16
1c00a7c4:	c298                	sw	a4,0(a3)
1c00a7c6:	01478693          	addi	a3,a5,20
1c00a7ca:	c298                	sw	a4,0(a3)
1c00a7cc:	01878693          	addi	a3,a5,24
1c00a7d0:	c298                	sw	a4,0(a3)
1c00a7d2:	01c78693          	addi	a3,a5,28
1c00a7d6:	c298                	sw	a4,0(a3)
1c00a7d8:	02078793          	addi	a5,a5,32
1c00a7dc:	1c0095b7          	lui	a1,0x1c009
1c00a7e0:	c398                	sw	a4,0(a5)
1c00a7e2:	e5658593          	addi	a1,a1,-426 # 1c008e56 <__rt_fc_socevents_handler>
1c00a7e6:	4569                	li	a0,26
1c00a7e8:	22ad                	jal	1c00a952 <rt_irq_set_handler>
1c00a7ea:	477d                	li	a4,31
1c00a7ec:	f14027f3          	csrr	a5,mhartid
1c00a7f0:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a7f4:	0ce79163          	bne	a5,a4,1c00a8b6 <__rt_init+0x144>
1c00a7f8:	1a1097b7          	lui	a5,0x1a109
1c00a7fc:	04000737          	lui	a4,0x4000
1c00a800:	c3d8                	sw	a4,4(a5)
1c00a802:	e36ff0ef          	jal	ra,1c009e38 <__rt_pmu_init>
1c00a806:	13b000ef          	jal	ra,1c00b140 <__rt_freq_init>
1c00a80a:	477d                	li	a4,31
1c00a80c:	f14027f3          	csrr	a5,mhartid
1c00a810:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a814:	0ae79763          	bne	a5,a4,1c00a8c2 <__rt_init+0x150>
1c00a818:	1a1097b7          	lui	a5,0x1a109
1c00a81c:	577d                	li	a4,-1
1c00a81e:	80e7a023          	sw	a4,-2048(a5) # 1a108800 <__l1_end+0xa1047e8>
1c00a822:	1c000437          	lui	s0,0x1c000
1c00a826:	262d                	jal	1c00ab50 <__rt_utils_init>
1c00a828:	57840413          	addi	s0,s0,1400 # 1c000578 <ctor_list+0x4>
1c00a82c:	a12ff0ef          	jal	ra,1c009a3e <__rt_allocs_init>
1c00a830:	718000ef          	jal	ra,1c00af48 <__rt_thread_sched_init>
1c00a834:	f2dfe0ef          	jal	ra,1c009760 <__rt_event_sched_init>
1c00a838:	143000ef          	jal	ra,1c00b17a <__rt_padframe_init>
1c00a83c:	0044278b          	p.lw	a5,4(s0!)
1c00a840:	e7d9                	bnez	a5,1c00a8ce <__rt_init+0x15c>
1c00a842:	30045073          	csrwi	mstatus,8
1c00a846:	4501                	li	a0,0
1c00a848:	2ce1                	jal	1c00ab20 <__rt_cbsys_exec>
1c00a84a:	e531                	bnez	a0,1c00a896 <__rt_init+0x124>
1c00a84c:	f14027f3          	csrr	a5,mhartid
1c00a850:	8795                	srai	a5,a5,0x5
1c00a852:	f267b7b3          	p.bclr	a5,a5,25,6
1c00a856:	477d                	li	a4,31
1c00a858:	0ae78d63          	beq	a5,a4,1c00a912 <__rt_init+0x1a0>
1c00a85c:	4681                	li	a3,0
1c00a85e:	4601                	li	a2,0
1c00a860:	4581                	li	a1,0
1c00a862:	4505                	li	a0,1
1c00a864:	c7bd                	beqz	a5,1c00a8d2 <__rt_init+0x160>
1c00a866:	3e79                	jal	1c00a404 <rt_cluster_mount>
1c00a868:	6595                	lui	a1,0x5
1c00a86a:	80058593          	addi	a1,a1,-2048 # 4800 <_l1_preload_size+0x7f0>
1c00a86e:	450d                	li	a0,3
1c00a870:	90eff0ef          	jal	ra,1c00997e <rt_alloc>
1c00a874:	872a                	mv	a4,a0
1c00a876:	c105                	beqz	a0,1c00a896 <__rt_init+0x124>
1c00a878:	6805                	lui	a6,0x1
1c00a87a:	80080813          	addi	a6,a6,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c00a87e:	1c00a637          	lui	a2,0x1c00a
1c00a882:	c002                	sw	zero,0(sp)
1c00a884:	48a5                	li	a7,9
1c00a886:	87c2                	mv	a5,a6
1c00a888:	4681                	li	a3,0
1c00a88a:	74e60613          	addi	a2,a2,1870 # 1c00a74e <cluster_start>
1c00a88e:	4581                	li	a1,0
1c00a890:	4501                	li	a0,0
1c00a892:	34cd                	jal	1c00a374 <rt_cluster_call>
1c00a894:	cd3d                	beqz	a0,1c00a912 <__rt_init+0x1a0>
1c00a896:	f1402673          	csrr	a2,mhartid
1c00a89a:	1c001537          	lui	a0,0x1c001
1c00a89e:	40565593          	srai	a1,a2,0x5
1c00a8a2:	f265b5b3          	p.bclr	a1,a1,25,6
1c00a8a6:	f4563633          	p.bclr	a2,a2,26,5
1c00a8aa:	adc50513          	addi	a0,a0,-1316 # 1c000adc <PIo2+0x180>
1c00a8ae:	02e010ef          	jal	ra,1c00b8dc <printf>
1c00a8b2:	7b9000ef          	jal	ra,1c00b86a <abort>
1c00a8b6:	002047b7          	lui	a5,0x204
1c00a8ba:	04000737          	lui	a4,0x4000
1c00a8be:	cbd8                	sw	a4,20(a5)
1c00a8c0:	b789                	j	1c00a802 <__rt_init+0x90>
1c00a8c2:	002017b7          	lui	a5,0x201
1c00a8c6:	577d                	li	a4,-1
1c00a8c8:	40e7a023          	sw	a4,1024(a5) # 201400 <__l1_heap_size+0x1e5418>
1c00a8cc:	bf99                	j	1c00a822 <__rt_init+0xb0>
1c00a8ce:	9782                	jalr	a5
1c00a8d0:	b7b5                	j	1c00a83c <__rt_init+0xca>
1c00a8d2:	3e0d                	jal	1c00a404 <rt_cluster_mount>
1c00a8d4:	6591                	lui	a1,0x4
1c00a8d6:	450d                	li	a0,3
1c00a8d8:	8a6ff0ef          	jal	ra,1c00997e <rt_alloc>
1c00a8dc:	dd4d                	beqz	a0,1c00a896 <__rt_init+0x124>
1c00a8de:	00204737          	lui	a4,0x204
1c00a8e2:	1ff00793          	li	a5,511
1c00a8e6:	08f72223          	sw	a5,132(a4) # 204084 <__l1_heap_size+0x1e809c>
1c00a8ea:	1c0107b7          	lui	a5,0x1c010
1c00a8ee:	15678793          	addi	a5,a5,342 # 1c010156 <__rt_set_slave_stack>
1c00a8f2:	c007c7b3          	p.bset	a5,a5,0,0
1c00a8f6:	08f72023          	sw	a5,128(a4)
1c00a8fa:	6785                	lui	a5,0x1
1c00a8fc:	80078793          	addi	a5,a5,-2048 # 800 <__rt_hyper_pending_tasks_last+0x298>
1c00a900:	08f72023          	sw	a5,128(a4)
1c00a904:	08a72023          	sw	a0,128(a4)
1c00a908:	4462                	lw	s0,24(sp)
1c00a90a:	40f2                	lw	ra,28(sp)
1c00a90c:	4501                	li	a0,0
1c00a90e:	6105                	addi	sp,sp,32
1c00a910:	bd3d                	j	1c00a74e <cluster_start>
1c00a912:	40f2                	lw	ra,28(sp)
1c00a914:	4462                	lw	s0,24(sp)
1c00a916:	6105                	addi	sp,sp,32
1c00a918:	8082                	ret

1c00a91a <__rt_deinit>:
1c00a91a:	1c0017b7          	lui	a5,0x1c001
1c00a91e:	6447a783          	lw	a5,1604(a5) # 1c001644 <__rt_platform>
1c00a922:	1141                	addi	sp,sp,-16
1c00a924:	c606                	sw	ra,12(sp)
1c00a926:	c422                	sw	s0,8(sp)
1c00a928:	0037b463          	p.bneimm	a5,3,1c00a930 <__rt_deinit+0x16>
1c00a92c:	7d005073          	csrwi	0x7d0,0
1c00a930:	4505                	li	a0,1
1c00a932:	1c000437          	lui	s0,0x1c000
1c00a936:	22ed                	jal	1c00ab20 <__rt_cbsys_exec>
1c00a938:	5b040413          	addi	s0,s0,1456 # 1c0005b0 <dtor_list+0x4>
1c00a93c:	0044278b          	p.lw	a5,4(s0!)
1c00a940:	e789                	bnez	a5,1c00a94a <__rt_deinit+0x30>
1c00a942:	40b2                	lw	ra,12(sp)
1c00a944:	4422                	lw	s0,8(sp)
1c00a946:	0141                	addi	sp,sp,16
1c00a948:	8082                	ret
1c00a94a:	9782                	jalr	a5
1c00a94c:	bfc5                	j	1c00a93c <__rt_deinit+0x22>

1c00a94e <pi_open_from_conf>:
1c00a94e:	c14c                	sw	a1,4(a0)
1c00a950:	8082                	ret

1c00a952 <rt_irq_set_handler>:
1c00a952:	f14027f3          	csrr	a5,mhartid
1c00a956:	477d                	li	a4,31
1c00a958:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a95c:	02e79e63          	bne	a5,a4,1c00a998 <rt_irq_set_handler+0x46>
1c00a960:	30502773          	csrr	a4,mtvec
1c00a964:	c0073733          	p.bclr	a4,a4,0,0
1c00a968:	050a                	slli	a0,a0,0x2
1c00a96a:	8d89                	sub	a1,a1,a0
1c00a96c:	8d99                	sub	a1,a1,a4
1c00a96e:	c14586b3          	p.extract	a3,a1,0,20
1c00a972:	06f00793          	li	a5,111
1c00a976:	c1f6a7b3          	p.insert	a5,a3,0,31
1c00a97a:	d21586b3          	p.extract	a3,a1,9,1
1c00a97e:	d356a7b3          	p.insert	a5,a3,9,21
1c00a982:	c0b586b3          	p.extract	a3,a1,0,11
1c00a986:	c146a7b3          	p.insert	a5,a3,0,20
1c00a98a:	cec585b3          	p.extract	a1,a1,7,12
1c00a98e:	cec5a7b3          	p.insert	a5,a1,7,12
1c00a992:	00f56723          	p.sw	a5,a4(a0)
1c00a996:	8082                	ret
1c00a998:	002007b7          	lui	a5,0x200
1c00a99c:	43b8                	lw	a4,64(a5)
1c00a99e:	b7e9                	j	1c00a968 <rt_irq_set_handler+0x16>

1c00a9a0 <illegal_insn_handler_c>:
1c00a9a0:	8082                	ret

1c00a9a2 <__rt_handle_illegal_instr>:
1c00a9a2:	1c0017b7          	lui	a5,0x1c001
1c00a9a6:	47c7a703          	lw	a4,1148(a5) # 1c00147c <__rt_debug_config>
1c00a9aa:	1141                	addi	sp,sp,-16
1c00a9ac:	c422                	sw	s0,8(sp)
1c00a9ae:	c606                	sw	ra,12(sp)
1c00a9b0:	fc173733          	p.bclr	a4,a4,30,1
1c00a9b4:	843e                	mv	s0,a5
1c00a9b6:	c315                	beqz	a4,1c00a9da <__rt_handle_illegal_instr+0x38>
1c00a9b8:	341026f3          	csrr	a3,mepc
1c00a9bc:	f1402673          	csrr	a2,mhartid
1c00a9c0:	1c001537          	lui	a0,0x1c001
1c00a9c4:	4298                	lw	a4,0(a3)
1c00a9c6:	40565593          	srai	a1,a2,0x5
1c00a9ca:	f265b5b3          	p.bclr	a1,a1,25,6
1c00a9ce:	f4563633          	p.bclr	a2,a2,26,5
1c00a9d2:	b3450513          	addi	a0,a0,-1228 # 1c000b34 <PIo2+0x1d8>
1c00a9d6:	707000ef          	jal	ra,1c00b8dc <printf>
1c00a9da:	47c42783          	lw	a5,1148(s0)
1c00a9de:	c01797b3          	p.extractu	a5,a5,0,1
1c00a9e2:	c399                	beqz	a5,1c00a9e8 <__rt_handle_illegal_instr+0x46>
1c00a9e4:	687000ef          	jal	ra,1c00b86a <abort>
1c00a9e8:	4422                	lw	s0,8(sp)
1c00a9ea:	40b2                	lw	ra,12(sp)
1c00a9ec:	0141                	addi	sp,sp,16
1c00a9ee:	fb3ff06f          	j	1c00a9a0 <illegal_insn_handler_c>

1c00a9f2 <__rt_irq_init>:
1c00a9f2:	f14027f3          	csrr	a5,mhartid
1c00a9f6:	477d                	li	a4,31
1c00a9f8:	ca5797b3          	p.extractu	a5,a5,5,5
1c00a9fc:	02e79463          	bne	a5,a4,1c00aa24 <__rt_irq_init+0x32>
1c00aa00:	1a1097b7          	lui	a5,0x1a109
1c00aa04:	577d                	li	a4,-1
1c00aa06:	c798                	sw	a4,8(a5)
1c00aa08:	f14027f3          	csrr	a5,mhartid
1c00aa0c:	477d                	li	a4,31
1c00aa0e:	ca5797b3          	p.extractu	a5,a5,5,5
1c00aa12:	00e79e63          	bne	a5,a4,1c00aa2e <__rt_irq_init+0x3c>
1c00aa16:	1c0087b7          	lui	a5,0x1c008
1c00aa1a:	00078793          	mv	a5,a5
1c00aa1e:	30579073          	csrw	mtvec,a5
1c00aa22:	8082                	ret
1c00aa24:	002047b7          	lui	a5,0x204
1c00aa28:	577d                	li	a4,-1
1c00aa2a:	cb98                	sw	a4,16(a5)
1c00aa2c:	bff1                	j	1c00aa08 <__rt_irq_init+0x16>
1c00aa2e:	1c0087b7          	lui	a5,0x1c008
1c00aa32:	00200737          	lui	a4,0x200
1c00aa36:	00078793          	mv	a5,a5
1c00aa3a:	c33c                	sw	a5,64(a4)
1c00aa3c:	8082                	ret

1c00aa3e <__rt_fc_cluster_lock_req>:
1c00aa3e:	1141                	addi	sp,sp,-16
1c00aa40:	c606                	sw	ra,12(sp)
1c00aa42:	c422                	sw	s0,8(sp)
1c00aa44:	c226                	sw	s1,4(sp)
1c00aa46:	300474f3          	csrrci	s1,mstatus,8
1c00aa4a:	09654703          	lbu	a4,150(a0)
1c00aa4e:	411c                	lw	a5,0(a0)
1c00aa50:	c721                	beqz	a4,1c00aa98 <__rt_fc_cluster_lock_req+0x5a>
1c00aa52:	4398                	lw	a4,0(a5)
1c00aa54:	c30d                	beqz	a4,1c00aa76 <__rt_fc_cluster_lock_req+0x38>
1c00aa56:	43d8                	lw	a4,4(a5)
1c00aa58:	cf09                	beqz	a4,1c00aa72 <__rt_fc_cluster_lock_req+0x34>
1c00aa5a:	4798                	lw	a4,8(a5)
1c00aa5c:	c348                	sw	a0,4(a4)
1c00aa5e:	c788                	sw	a0,8(a5)
1c00aa60:	00052223          	sw	zero,4(a0)
1c00aa64:	30049073          	csrw	mstatus,s1
1c00aa68:	40b2                	lw	ra,12(sp)
1c00aa6a:	4422                	lw	s0,8(sp)
1c00aa6c:	4492                	lw	s1,4(sp)
1c00aa6e:	0141                	addi	sp,sp,16
1c00aa70:	8082                	ret
1c00aa72:	c3c8                	sw	a0,4(a5)
1c00aa74:	b7ed                	j	1c00aa5e <__rt_fc_cluster_lock_req+0x20>
1c00aa76:	4705                	li	a4,1
1c00aa78:	08e50a23          	sb	a4,148(a0)
1c00aa7c:	4705                	li	a4,1
1c00aa7e:	c398                	sw	a4,0(a5)
1c00aa80:	09554783          	lbu	a5,149(a0)
1c00aa84:	04078793          	addi	a5,a5,64 # 1c008040 <__irq_vector_base+0x40>
1c00aa88:	00201737          	lui	a4,0x201
1c00aa8c:	07da                	slli	a5,a5,0x16
1c00aa8e:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e4e1c>
1c00aa92:	0007e723          	p.sw	zero,a4(a5)
1c00aa96:	b7f9                	j	1c00aa64 <__rt_fc_cluster_lock_req+0x26>
1c00aa98:	842a                	mv	s0,a0
1c00aa9a:	47c8                	lw	a0,12(a5)
1c00aa9c:	cd01                	beqz	a0,1c00aab4 <__rt_fc_cluster_lock_req+0x76>
1c00aa9e:	0007a023          	sw	zero,0(a5)
1c00aaa2:	0007a623          	sw	zero,12(a5)
1c00aaa6:	2171                	jal	1c00af32 <__rt_thread_wakeup>
1c00aaa8:	4785                	li	a5,1
1c00aaaa:	08f40a23          	sb	a5,148(s0)
1c00aaae:	09544783          	lbu	a5,149(s0)
1c00aab2:	bfc9                	j	1c00aa84 <__rt_fc_cluster_lock_req+0x46>
1c00aab4:	43d8                	lw	a4,4(a5)
1c00aab6:	e701                	bnez	a4,1c00aabe <__rt_fc_cluster_lock_req+0x80>
1c00aab8:	0007a023          	sw	zero,0(a5)
1c00aabc:	b7f5                	j	1c00aaa8 <__rt_fc_cluster_lock_req+0x6a>
1c00aabe:	4354                	lw	a3,4(a4)
1c00aac0:	c3d4                	sw	a3,4(a5)
1c00aac2:	4785                	li	a5,1
1c00aac4:	08f70a23          	sb	a5,148(a4)
1c00aac8:	09574783          	lbu	a5,149(a4)
1c00aacc:	00201737          	lui	a4,0x201
1c00aad0:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e4e1c>
1c00aad4:	04078793          	addi	a5,a5,64
1c00aad8:	07da                	slli	a5,a5,0x16
1c00aada:	0007e723          	p.sw	zero,a4(a5)
1c00aade:	b7e9                	j	1c00aaa8 <__rt_fc_cluster_lock_req+0x6a>

1c00aae0 <__rt_cbsys_add>:
1c00aae0:	1101                	addi	sp,sp,-32
1c00aae2:	cc22                	sw	s0,24(sp)
1c00aae4:	ca26                	sw	s1,20(sp)
1c00aae6:	842a                	mv	s0,a0
1c00aae8:	84ae                	mv	s1,a1
1c00aaea:	4501                	li	a0,0
1c00aaec:	45b1                	li	a1,12
1c00aaee:	c632                	sw	a2,12(sp)
1c00aaf0:	ce06                	sw	ra,28(sp)
1c00aaf2:	e8dfe0ef          	jal	ra,1c00997e <rt_alloc>
1c00aaf6:	4632                	lw	a2,12(sp)
1c00aaf8:	c115                	beqz	a0,1c00ab1c <__rt_cbsys_add+0x3c>
1c00aafa:	1c0017b7          	lui	a5,0x1c001
1c00aafe:	040a                	slli	s0,s0,0x2
1c00ab00:	48078793          	addi	a5,a5,1152 # 1c001480 <cbsys_first>
1c00ab04:	97a2                	add	a5,a5,s0
1c00ab06:	4398                	lw	a4,0(a5)
1c00ab08:	c104                	sw	s1,0(a0)
1c00ab0a:	c150                	sw	a2,4(a0)
1c00ab0c:	c518                	sw	a4,8(a0)
1c00ab0e:	c388                	sw	a0,0(a5)
1c00ab10:	4501                	li	a0,0
1c00ab12:	40f2                	lw	ra,28(sp)
1c00ab14:	4462                	lw	s0,24(sp)
1c00ab16:	44d2                	lw	s1,20(sp)
1c00ab18:	6105                	addi	sp,sp,32
1c00ab1a:	8082                	ret
1c00ab1c:	557d                	li	a0,-1
1c00ab1e:	bfd5                	j	1c00ab12 <__rt_cbsys_add+0x32>

1c00ab20 <__rt_cbsys_exec>:
1c00ab20:	1141                	addi	sp,sp,-16
1c00ab22:	c422                	sw	s0,8(sp)
1c00ab24:	1c001437          	lui	s0,0x1c001
1c00ab28:	050a                	slli	a0,a0,0x2
1c00ab2a:	48040413          	addi	s0,s0,1152 # 1c001480 <cbsys_first>
1c00ab2e:	20a47403          	p.lw	s0,a0(s0)
1c00ab32:	c606                	sw	ra,12(sp)
1c00ab34:	e411                	bnez	s0,1c00ab40 <__rt_cbsys_exec+0x20>
1c00ab36:	4501                	li	a0,0
1c00ab38:	40b2                	lw	ra,12(sp)
1c00ab3a:	4422                	lw	s0,8(sp)
1c00ab3c:	0141                	addi	sp,sp,16
1c00ab3e:	8082                	ret
1c00ab40:	401c                	lw	a5,0(s0)
1c00ab42:	4048                	lw	a0,4(s0)
1c00ab44:	9782                	jalr	a5
1c00ab46:	e119                	bnez	a0,1c00ab4c <__rt_cbsys_exec+0x2c>
1c00ab48:	4400                	lw	s0,8(s0)
1c00ab4a:	b7ed                	j	1c00ab34 <__rt_cbsys_exec+0x14>
1c00ab4c:	557d                	li	a0,-1
1c00ab4e:	b7ed                	j	1c00ab38 <__rt_cbsys_exec+0x18>

1c00ab50 <__rt_utils_init>:
1c00ab50:	1c0017b7          	lui	a5,0x1c001
1c00ab54:	48078793          	addi	a5,a5,1152 # 1c001480 <cbsys_first>
1c00ab58:	0007a023          	sw	zero,0(a5)
1c00ab5c:	0007a223          	sw	zero,4(a5)
1c00ab60:	0007a423          	sw	zero,8(a5)
1c00ab64:	0007a623          	sw	zero,12(a5)
1c00ab68:	0007a823          	sw	zero,16(a5)
1c00ab6c:	0007aa23          	sw	zero,20(a5)
1c00ab70:	8082                	ret

1c00ab72 <__rt_fc_lock>:
1c00ab72:	1141                	addi	sp,sp,-16
1c00ab74:	c422                	sw	s0,8(sp)
1c00ab76:	842a                	mv	s0,a0
1c00ab78:	c606                	sw	ra,12(sp)
1c00ab7a:	c226                	sw	s1,4(sp)
1c00ab7c:	c04a                	sw	s2,0(sp)
1c00ab7e:	300474f3          	csrrci	s1,mstatus,8
1c00ab82:	401c                	lw	a5,0(s0)
1c00ab84:	eb99                	bnez	a5,1c00ab9a <__rt_fc_lock+0x28>
1c00ab86:	4785                	li	a5,1
1c00ab88:	c01c                	sw	a5,0(s0)
1c00ab8a:	30049073          	csrw	mstatus,s1
1c00ab8e:	40b2                	lw	ra,12(sp)
1c00ab90:	4422                	lw	s0,8(sp)
1c00ab92:	4492                	lw	s1,4(sp)
1c00ab94:	4902                	lw	s2,0(sp)
1c00ab96:	0141                	addi	sp,sp,16
1c00ab98:	8082                	ret
1c00ab9a:	04802783          	lw	a5,72(zero) # 48 <__rt_thread_current>
1c00ab9e:	4585                	li	a1,1
1c00aba0:	e3ff5517          	auipc	a0,0xe3ff5
1c00aba4:	46c50513          	addi	a0,a0,1132 # c <__rt_sched>
1c00aba8:	c45c                	sw	a5,12(s0)
1c00abaa:	b15fe0ef          	jal	ra,1c0096be <__rt_event_execute>
1c00abae:	bfd1                	j	1c00ab82 <__rt_fc_lock+0x10>

1c00abb0 <__rt_fc_unlock>:
1c00abb0:	300476f3          	csrrci	a3,mstatus,8
1c00abb4:	415c                	lw	a5,4(a0)
1c00abb6:	e791                	bnez	a5,1c00abc2 <__rt_fc_unlock+0x12>
1c00abb8:	00052023          	sw	zero,0(a0)
1c00abbc:	30069073          	csrw	mstatus,a3
1c00abc0:	8082                	ret
1c00abc2:	43d8                	lw	a4,4(a5)
1c00abc4:	c158                	sw	a4,4(a0)
1c00abc6:	4705                	li	a4,1
1c00abc8:	08e78a23          	sb	a4,148(a5)
1c00abcc:	0957c783          	lbu	a5,149(a5)
1c00abd0:	00201737          	lui	a4,0x201
1c00abd4:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e4e1c>
1c00abd8:	04078793          	addi	a5,a5,64
1c00abdc:	07da                	slli	a5,a5,0x16
1c00abde:	0007e723          	p.sw	zero,a4(a5)
1c00abe2:	bfe9                	j	1c00abbc <__rt_fc_unlock+0xc>

1c00abe4 <__rt_fc_cluster_lock>:
1c00abe4:	f14027f3          	csrr	a5,mhartid
1c00abe8:	8795                	srai	a5,a5,0x5
1c00abea:	f267b7b3          	p.bclr	a5,a5,25,6
1c00abee:	08f58aa3          	sb	a5,149(a1) # 4095 <_l1_preload_size+0x85>
1c00abf2:	4785                	li	a5,1
1c00abf4:	08f58b23          	sb	a5,150(a1)
1c00abf8:	1c00b7b7          	lui	a5,0x1c00b
1c00abfc:	a3e78793          	addi	a5,a5,-1474 # 1c00aa3e <__rt_fc_cluster_lock_req>
1c00ac00:	c188                	sw	a0,0(a1)
1c00ac02:	08058a23          	sb	zero,148(a1)
1c00ac06:	0205a423          	sw	zero,40(a1)
1c00ac0a:	0205a623          	sw	zero,44(a1)
1c00ac0e:	c59c                	sw	a5,8(a1)
1c00ac10:	c5cc                	sw	a1,12(a1)
1c00ac12:	05a1                	addi	a1,a1,8
1c00ac14:	c005c533          	p.bset	a0,a1,0,0
1c00ac18:	e96ff06f          	j	1c00a2ae <__rt_cluster_push_fc_event>

1c00ac1c <__rt_fc_cluster_unlock>:
1c00ac1c:	f14027f3          	csrr	a5,mhartid
1c00ac20:	8795                	srai	a5,a5,0x5
1c00ac22:	f267b7b3          	p.bclr	a5,a5,25,6
1c00ac26:	08f58aa3          	sb	a5,149(a1)
1c00ac2a:	1c00b7b7          	lui	a5,0x1c00b
1c00ac2e:	a3e78793          	addi	a5,a5,-1474 # 1c00aa3e <__rt_fc_cluster_lock_req>
1c00ac32:	c188                	sw	a0,0(a1)
1c00ac34:	08058a23          	sb	zero,148(a1)
1c00ac38:	08058b23          	sb	zero,150(a1)
1c00ac3c:	0205a423          	sw	zero,40(a1)
1c00ac40:	0205a623          	sw	zero,44(a1)
1c00ac44:	c59c                	sw	a5,8(a1)
1c00ac46:	c5cc                	sw	a1,12(a1)
1c00ac48:	05a1                	addi	a1,a1,8
1c00ac4a:	c005c533          	p.bset	a0,a1,0,0
1c00ac4e:	e60ff06f          	j	1c00a2ae <__rt_cluster_push_fc_event>

1c00ac52 <__rt_event_enqueue>:
1c00ac52:	00c02783          	lw	a5,12(zero) # c <__rt_sched>
1c00ac56:	00052c23          	sw	zero,24(a0)
1c00ac5a:	c799                	beqz	a5,1c00ac68 <__rt_event_enqueue+0x16>
1c00ac5c:	01002783          	lw	a5,16(zero) # 10 <__rt_sched+0x4>
1c00ac60:	cf88                	sw	a0,24(a5)
1c00ac62:	00a02823          	sw	a0,16(zero) # 10 <__rt_sched+0x4>
1c00ac66:	8082                	ret
1c00ac68:	00a02623          	sw	a0,12(zero) # c <__rt_sched>
1c00ac6c:	bfdd                	j	1c00ac62 <__rt_event_enqueue+0x10>

1c00ac6e <__rt_bridge_check_bridge_req.part.5>:
1c00ac6e:	1c001737          	lui	a4,0x1c001
1c00ac72:	58470793          	addi	a5,a4,1412 # 1c001584 <__hal_debug_struct>
1c00ac76:	0a47a783          	lw	a5,164(a5)
1c00ac7a:	58470713          	addi	a4,a4,1412
1c00ac7e:	c789                	beqz	a5,1c00ac88 <__rt_bridge_check_bridge_req.part.5+0x1a>
1c00ac80:	4f94                	lw	a3,24(a5)
1c00ac82:	e681                	bnez	a3,1c00ac8a <__rt_bridge_check_bridge_req.part.5+0x1c>
1c00ac84:	0af72623          	sw	a5,172(a4)
1c00ac88:	8082                	ret
1c00ac8a:	479c                	lw	a5,8(a5)
1c00ac8c:	bfcd                	j	1c00ac7e <__rt_bridge_check_bridge_req.part.5+0x10>

1c00ac8e <__rt_bridge_wait>:
1c00ac8e:	f14027f3          	csrr	a5,mhartid
1c00ac92:	477d                	li	a4,31
1c00ac94:	ca5797b3          	p.extractu	a5,a5,5,5
1c00ac98:	02e79e63          	bne	a5,a4,1c00acd4 <__rt_bridge_wait+0x46>
1c00ac9c:	1a1097b7          	lui	a5,0x1a109
1c00aca0:	00c78513          	addi	a0,a5,12 # 1a10900c <__l1_end+0xa104ff4>
1c00aca4:	6711                	lui	a4,0x4
1c00aca6:	00478593          	addi	a1,a5,4
1c00acaa:	00878613          	addi	a2,a5,8
1c00acae:	300476f3          	csrrci	a3,mstatus,8
1c00acb2:	00052803          	lw	a6,0(a0)
1c00acb6:	01181893          	slli	a7,a6,0x11
1c00acba:	0008c963          	bltz	a7,1c00accc <__rt_bridge_wait+0x3e>
1c00acbe:	c198                	sw	a4,0(a1)
1c00acc0:	10500073          	wfi
1c00acc4:	c218                	sw	a4,0(a2)
1c00acc6:	30069073          	csrw	mstatus,a3
1c00acca:	b7d5                	j	1c00acae <__rt_bridge_wait+0x20>
1c00accc:	07d1                	addi	a5,a5,20
1c00acce:	c398                	sw	a4,0(a5)
1c00acd0:	30069073          	csrw	mstatus,a3
1c00acd4:	8082                	ret

1c00acd6 <__rt_bridge_handle_notif>:
1c00acd6:	1141                	addi	sp,sp,-16
1c00acd8:	c422                	sw	s0,8(sp)
1c00acda:	1c001437          	lui	s0,0x1c001
1c00acde:	58440793          	addi	a5,s0,1412 # 1c001584 <__hal_debug_struct>
1c00ace2:	0a47a783          	lw	a5,164(a5)
1c00ace6:	c606                	sw	ra,12(sp)
1c00ace8:	c226                	sw	s1,4(sp)
1c00acea:	c04a                	sw	s2,0(sp)
1c00acec:	58440413          	addi	s0,s0,1412
1c00acf0:	c399                	beqz	a5,1c00acf6 <__rt_bridge_handle_notif+0x20>
1c00acf2:	4bd8                	lw	a4,20(a5)
1c00acf4:	e30d                	bnez	a4,1c00ad16 <__rt_bridge_handle_notif+0x40>
1c00acf6:	0b442783          	lw	a5,180(s0)
1c00acfa:	c789                	beqz	a5,1c00ad04 <__rt_bridge_handle_notif+0x2e>
1c00acfc:	43a8                	lw	a0,64(a5)
1c00acfe:	0a042a23          	sw	zero,180(s0)
1c00ad02:	3f81                	jal	1c00ac52 <__rt_event_enqueue>
1c00ad04:	0ac42783          	lw	a5,172(s0)
1c00ad08:	eb95                	bnez	a5,1c00ad3c <__rt_bridge_handle_notif+0x66>
1c00ad0a:	4422                	lw	s0,8(sp)
1c00ad0c:	40b2                	lw	ra,12(sp)
1c00ad0e:	4492                	lw	s1,4(sp)
1c00ad10:	4902                	lw	s2,0(sp)
1c00ad12:	0141                	addi	sp,sp,16
1c00ad14:	bfa9                	j	1c00ac6e <__rt_bridge_check_bridge_req.part.5>
1c00ad16:	4784                	lw	s1,8(a5)
1c00ad18:	4fd8                	lw	a4,28(a5)
1c00ad1a:	0a942223          	sw	s1,164(s0)
1c00ad1e:	cb01                	beqz	a4,1c00ad2e <__rt_bridge_handle_notif+0x58>
1c00ad20:	0b042703          	lw	a4,176(s0)
1c00ad24:	c798                	sw	a4,8(a5)
1c00ad26:	0af42823          	sw	a5,176(s0)
1c00ad2a:	87a6                	mv	a5,s1
1c00ad2c:	b7d1                	j	1c00acf0 <__rt_bridge_handle_notif+0x1a>
1c00ad2e:	43a8                	lw	a0,64(a5)
1c00ad30:	30047973          	csrrci	s2,mstatus,8
1c00ad34:	3f39                	jal	1c00ac52 <__rt_event_enqueue>
1c00ad36:	30091073          	csrw	mstatus,s2
1c00ad3a:	bfc5                	j	1c00ad2a <__rt_bridge_handle_notif+0x54>
1c00ad3c:	40b2                	lw	ra,12(sp)
1c00ad3e:	4422                	lw	s0,8(sp)
1c00ad40:	4492                	lw	s1,4(sp)
1c00ad42:	4902                	lw	s2,0(sp)
1c00ad44:	0141                	addi	sp,sp,16
1c00ad46:	8082                	ret

1c00ad48 <__rt_bridge_check_connection>:
1c00ad48:	1c001737          	lui	a4,0x1c001
1c00ad4c:	58470713          	addi	a4,a4,1412 # 1c001584 <__hal_debug_struct>
1c00ad50:	471c                	lw	a5,8(a4)
1c00ad52:	ef8d                	bnez	a5,1c00ad8c <__rt_bridge_check_connection+0x44>
1c00ad54:	1a1047b7          	lui	a5,0x1a104
1c00ad58:	07478793          	addi	a5,a5,116 # 1a104074 <__l1_end+0xa10005c>
1c00ad5c:	4394                	lw	a3,0(a5)
1c00ad5e:	cc9696b3          	p.extractu	a3,a3,6,9
1c00ad62:	0276b563          	p.bneimm	a3,7,1c00ad8c <__rt_bridge_check_connection+0x44>
1c00ad66:	1141                	addi	sp,sp,-16
1c00ad68:	c422                	sw	s0,8(sp)
1c00ad6a:	c606                	sw	ra,12(sp)
1c00ad6c:	4685                	li	a3,1
1c00ad6e:	c714                	sw	a3,8(a4)
1c00ad70:	4709                	li	a4,2
1c00ad72:	c398                	sw	a4,0(a5)
1c00ad74:	843e                	mv	s0,a5
1c00ad76:	401c                	lw	a5,0(s0)
1c00ad78:	cc9797b3          	p.extractu	a5,a5,6,9
1c00ad7c:	0077a663          	p.beqimm	a5,7,1c00ad88 <__rt_bridge_check_connection+0x40>
1c00ad80:	40b2                	lw	ra,12(sp)
1c00ad82:	4422                	lw	s0,8(sp)
1c00ad84:	0141                	addi	sp,sp,16
1c00ad86:	8082                	ret
1c00ad88:	3719                	jal	1c00ac8e <__rt_bridge_wait>
1c00ad8a:	b7f5                	j	1c00ad76 <__rt_bridge_check_connection+0x2e>
1c00ad8c:	8082                	ret

1c00ad8e <__rt_bridge_set_available>:
1c00ad8e:	1c0017b7          	lui	a5,0x1c001
1c00ad92:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00ad96:	4798                	lw	a4,8(a5)
1c00ad98:	1a1047b7          	lui	a5,0x1a104
1c00ad9c:	07478793          	addi	a5,a5,116 # 1a104074 <__l1_end+0xa10005c>
1c00ada0:	e701                	bnez	a4,1c00ada8 <__rt_bridge_set_available+0x1a>
1c00ada2:	4721                	li	a4,8
1c00ada4:	c398                	sw	a4,0(a5)
1c00ada6:	8082                	ret
1c00ada8:	4709                	li	a4,2
1c00adaa:	bfed                	j	1c00ada4 <__rt_bridge_set_available+0x16>

1c00adac <__rt_bridge_send_notif>:
1c00adac:	1141                	addi	sp,sp,-16
1c00adae:	c606                	sw	ra,12(sp)
1c00adb0:	3f61                	jal	1c00ad48 <__rt_bridge_check_connection>
1c00adb2:	1c0017b7          	lui	a5,0x1c001
1c00adb6:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00adba:	479c                	lw	a5,8(a5)
1c00adbc:	c789                	beqz	a5,1c00adc6 <__rt_bridge_send_notif+0x1a>
1c00adbe:	1a1047b7          	lui	a5,0x1a104
1c00adc2:	4719                	li	a4,6
1c00adc4:	dbf8                	sw	a4,116(a5)
1c00adc6:	40b2                	lw	ra,12(sp)
1c00adc8:	0141                	addi	sp,sp,16
1c00adca:	8082                	ret

1c00adcc <__rt_bridge_clear_notif>:
1c00adcc:	1141                	addi	sp,sp,-16
1c00adce:	c606                	sw	ra,12(sp)
1c00add0:	3fa5                	jal	1c00ad48 <__rt_bridge_check_connection>
1c00add2:	1c0017b7          	lui	a5,0x1c001
1c00add6:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00adda:	479c                	lw	a5,8(a5)
1c00addc:	c781                	beqz	a5,1c00ade4 <__rt_bridge_clear_notif+0x18>
1c00adde:	40b2                	lw	ra,12(sp)
1c00ade0:	0141                	addi	sp,sp,16
1c00ade2:	b775                	j	1c00ad8e <__rt_bridge_set_available>
1c00ade4:	40b2                	lw	ra,12(sp)
1c00ade6:	0141                	addi	sp,sp,16
1c00ade8:	8082                	ret

1c00adea <__rt_bridge_printf_flush>:
1c00adea:	1141                	addi	sp,sp,-16
1c00adec:	c422                	sw	s0,8(sp)
1c00adee:	c606                	sw	ra,12(sp)
1c00adf0:	1c001437          	lui	s0,0x1c001
1c00adf4:	3f91                	jal	1c00ad48 <__rt_bridge_check_connection>
1c00adf6:	58440793          	addi	a5,s0,1412 # 1c001584 <__hal_debug_struct>
1c00adfa:	479c                	lw	a5,8(a5)
1c00adfc:	c385                	beqz	a5,1c00ae1c <__rt_bridge_printf_flush+0x32>
1c00adfe:	58440413          	addi	s0,s0,1412
1c00ae02:	485c                	lw	a5,20(s0)
1c00ae04:	e399                	bnez	a5,1c00ae0a <__rt_bridge_printf_flush+0x20>
1c00ae06:	4c1c                	lw	a5,24(s0)
1c00ae08:	cb91                	beqz	a5,1c00ae1c <__rt_bridge_printf_flush+0x32>
1c00ae0a:	374d                	jal	1c00adac <__rt_bridge_send_notif>
1c00ae0c:	485c                	lw	a5,20(s0)
1c00ae0e:	e789                	bnez	a5,1c00ae18 <__rt_bridge_printf_flush+0x2e>
1c00ae10:	4422                	lw	s0,8(sp)
1c00ae12:	40b2                	lw	ra,12(sp)
1c00ae14:	0141                	addi	sp,sp,16
1c00ae16:	bf5d                	j	1c00adcc <__rt_bridge_clear_notif>
1c00ae18:	3d9d                	jal	1c00ac8e <__rt_bridge_wait>
1c00ae1a:	bfcd                	j	1c00ae0c <__rt_bridge_printf_flush+0x22>
1c00ae1c:	40b2                	lw	ra,12(sp)
1c00ae1e:	4422                	lw	s0,8(sp)
1c00ae20:	0141                	addi	sp,sp,16
1c00ae22:	8082                	ret

1c00ae24 <__rt_bridge_req_shutdown>:
1c00ae24:	1141                	addi	sp,sp,-16
1c00ae26:	c606                	sw	ra,12(sp)
1c00ae28:	c422                	sw	s0,8(sp)
1c00ae2a:	3f39                	jal	1c00ad48 <__rt_bridge_check_connection>
1c00ae2c:	1c0017b7          	lui	a5,0x1c001
1c00ae30:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00ae34:	479c                	lw	a5,8(a5)
1c00ae36:	c3a9                	beqz	a5,1c00ae78 <__rt_bridge_req_shutdown+0x54>
1c00ae38:	1a104437          	lui	s0,0x1a104
1c00ae3c:	377d                	jal	1c00adea <__rt_bridge_printf_flush>
1c00ae3e:	07440413          	addi	s0,s0,116 # 1a104074 <__l1_end+0xa10005c>
1c00ae42:	401c                	lw	a5,0(s0)
1c00ae44:	cc9797b3          	p.extractu	a5,a5,6,9
1c00ae48:	0277ac63          	p.beqimm	a5,7,1c00ae80 <__rt_bridge_req_shutdown+0x5c>
1c00ae4c:	4791                	li	a5,4
1c00ae4e:	c01c                	sw	a5,0(s0)
1c00ae50:	1a104437          	lui	s0,0x1a104
1c00ae54:	07440413          	addi	s0,s0,116 # 1a104074 <__l1_end+0xa10005c>
1c00ae58:	401c                	lw	a5,0(s0)
1c00ae5a:	cc9797b3          	p.extractu	a5,a5,6,9
1c00ae5e:	0277b363          	p.bneimm	a5,7,1c00ae84 <__rt_bridge_req_shutdown+0x60>
1c00ae62:	00042023          	sw	zero,0(s0)
1c00ae66:	1a104437          	lui	s0,0x1a104
1c00ae6a:	07440413          	addi	s0,s0,116 # 1a104074 <__l1_end+0xa10005c>
1c00ae6e:	401c                	lw	a5,0(s0)
1c00ae70:	cc9797b3          	p.extractu	a5,a5,6,9
1c00ae74:	0077aa63          	p.beqimm	a5,7,1c00ae88 <__rt_bridge_req_shutdown+0x64>
1c00ae78:	40b2                	lw	ra,12(sp)
1c00ae7a:	4422                	lw	s0,8(sp)
1c00ae7c:	0141                	addi	sp,sp,16
1c00ae7e:	8082                	ret
1c00ae80:	3539                	jal	1c00ac8e <__rt_bridge_wait>
1c00ae82:	b7c1                	j	1c00ae42 <__rt_bridge_req_shutdown+0x1e>
1c00ae84:	3529                	jal	1c00ac8e <__rt_bridge_wait>
1c00ae86:	bfc9                	j	1c00ae58 <__rt_bridge_req_shutdown+0x34>
1c00ae88:	3519                	jal	1c00ac8e <__rt_bridge_wait>
1c00ae8a:	b7d5                	j	1c00ae6e <__rt_bridge_req_shutdown+0x4a>

1c00ae8c <__rt_bridge_init>:
1c00ae8c:	1c0017b7          	lui	a5,0x1c001
1c00ae90:	1a109737          	lui	a4,0x1a109
1c00ae94:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00ae98:	0741                	addi	a4,a4,16
1c00ae9a:	0ae7ac23          	sw	a4,184(a5)
1c00ae9e:	4741                	li	a4,16
1c00aea0:	0a07a223          	sw	zero,164(a5)
1c00aea4:	0a07a623          	sw	zero,172(a5)
1c00aea8:	0a07aa23          	sw	zero,180(a5)
1c00aeac:	0ae7ae23          	sw	a4,188(a5)
1c00aeb0:	02c00793          	li	a5,44
1c00aeb4:	0007a823          	sw	zero,16(a5)
1c00aeb8:	0007a023          	sw	zero,0(a5)
1c00aebc:	8082                	ret

1c00aebe <__rt_thread_enqueue_ready>:
1c00aebe:	04002703          	lw	a4,64(zero) # 40 <__rt_ready_queue>
1c00aec2:	02052c23          	sw	zero,56(a0)
1c00aec6:	04000793          	li	a5,64
1c00aeca:	e711                	bnez	a4,1c00aed6 <__rt_thread_enqueue_ready+0x18>
1c00aecc:	c388                	sw	a0,0(a5)
1c00aece:	c3c8                	sw	a0,4(a5)
1c00aed0:	0c052a23          	sw	zero,212(a0)
1c00aed4:	8082                	ret
1c00aed6:	43d8                	lw	a4,4(a5)
1c00aed8:	df08                	sw	a0,56(a4)
1c00aeda:	bfd5                	j	1c00aece <__rt_thread_enqueue_ready+0x10>

1c00aedc <__rt_thread_sleep>:
1c00aedc:	04000713          	li	a4,64
1c00aee0:	4708                	lw	a0,8(a4)
1c00aee2:	04000793          	li	a5,64
1c00aee6:	438c                	lw	a1,0(a5)
1c00aee8:	c999                	beqz	a1,1c00aefe <__rt_thread_sleep+0x22>
1c00aeea:	5d98                	lw	a4,56(a1)
1c00aeec:	c398                	sw	a4,0(a5)
1c00aeee:	4705                	li	a4,1
1c00aef0:	0ce5aa23          	sw	a4,212(a1)
1c00aef4:	00b50c63          	beq	a0,a1,1c00af0c <__rt_thread_sleep+0x30>
1c00aef8:	c78c                	sw	a1,8(a5)
1c00aefa:	ef3fd06f          	j	1c008dec <__rt_thread_switch>
1c00aefe:	10500073          	wfi
1c00af02:	30045073          	csrwi	mstatus,8
1c00af06:	30047773          	csrrci	a4,mstatus,8
1c00af0a:	bff1                	j	1c00aee6 <__rt_thread_sleep+0xa>
1c00af0c:	8082                	ret

1c00af0e <rt_thread_exit>:
1c00af0e:	300477f3          	csrrci	a5,mstatus,8
1c00af12:	04802783          	lw	a5,72(zero) # 48 <__rt_thread_current>
1c00af16:	4705                	li	a4,1
1c00af18:	c3e8                	sw	a0,68(a5)
1c00af1a:	5fc8                	lw	a0,60(a5)
1c00af1c:	c3b8                	sw	a4,64(a5)
1c00af1e:	c909                	beqz	a0,1c00af30 <rt_thread_exit+0x22>
1c00af20:	0d452783          	lw	a5,212(a0)
1c00af24:	c791                	beqz	a5,1c00af30 <rt_thread_exit+0x22>
1c00af26:	1141                	addi	sp,sp,-16
1c00af28:	c606                	sw	ra,12(sp)
1c00af2a:	3f51                	jal	1c00aebe <__rt_thread_enqueue_ready>
1c00af2c:	40b2                	lw	ra,12(sp)
1c00af2e:	0141                	addi	sp,sp,16
1c00af30:	b775                	j	1c00aedc <__rt_thread_sleep>

1c00af32 <__rt_thread_wakeup>:
1c00af32:	5d18                	lw	a4,56(a0)
1c00af34:	eb09                	bnez	a4,1c00af46 <__rt_thread_wakeup+0x14>
1c00af36:	04802703          	lw	a4,72(zero) # 48 <__rt_thread_current>
1c00af3a:	00a70663          	beq	a4,a0,1c00af46 <__rt_thread_wakeup+0x14>
1c00af3e:	0d452783          	lw	a5,212(a0)
1c00af42:	c391                	beqz	a5,1c00af46 <__rt_thread_wakeup+0x14>
1c00af44:	bfad                	j	1c00aebe <__rt_thread_enqueue_ready>
1c00af46:	8082                	ret

1c00af48 <__rt_thread_sched_init>:
1c00af48:	1141                	addi	sp,sp,-16
1c00af4a:	c422                	sw	s0,8(sp)
1c00af4c:	1c0097b7          	lui	a5,0x1c009
1c00af50:	1c001437          	lui	s0,0x1c001
1c00af54:	c226                	sw	s1,4(sp)
1c00af56:	c04a                	sw	s2,0(sp)
1c00af58:	c606                	sw	ra,12(sp)
1c00af5a:	49840413          	addi	s0,s0,1176 # 1c001498 <__rt_thread_main>
1c00af5e:	de678793          	addi	a5,a5,-538 # 1c008de6 <__rt_thread_start>
1c00af62:	c01c                	sw	a5,0(s0)
1c00af64:	1c00b7b7          	lui	a5,0x1c00b
1c00af68:	04840913          	addi	s2,s0,72
1c00af6c:	f0e78793          	addi	a5,a5,-242 # 1c00af0e <rt_thread_exit>
1c00af70:	04000493          	li	s1,64
1c00af74:	c45c                	sw	a5,12(s0)
1c00af76:	854a                	mv	a0,s2
1c00af78:	4785                	li	a5,1
1c00af7a:	e3ff5597          	auipc	a1,0xe3ff5
1c00af7e:	09258593          	addi	a1,a1,146 # c <__rt_sched>
1c00af82:	0cf42a23          	sw	a5,212(s0)
1c00af86:	0004a023          	sw	zero,0(s1)
1c00af8a:	02042a23          	sw	zero,52(s0)
1c00af8e:	00042223          	sw	zero,4(s0)
1c00af92:	00042423          	sw	zero,8(s0)
1c00af96:	e28fe0ef          	jal	ra,1c0095be <__rt_event_init>
1c00af9a:	00802783          	lw	a5,8(zero) # 8 <__rt_first_free>
1c00af9e:	c480                	sw	s0,8(s1)
1c00afa0:	40b2                	lw	ra,12(sp)
1c00afa2:	d03c                	sw	a5,96(s0)
1c00afa4:	4422                	lw	s0,8(sp)
1c00afa6:	01202423          	sw	s2,8(zero) # 8 <__rt_first_free>
1c00afaa:	4492                	lw	s1,4(sp)
1c00afac:	4902                	lw	s2,0(sp)
1c00afae:	0141                	addi	sp,sp,16
1c00afb0:	8082                	ret

1c00afb2 <__rt_fll_set_freq>:
1c00afb2:	1101                	addi	sp,sp,-32
1c00afb4:	cc22                	sw	s0,24(sp)
1c00afb6:	ce06                	sw	ra,28(sp)
1c00afb8:	842a                	mv	s0,a0
1c00afba:	00153563          	p.bneimm	a0,1,1c00afc4 <__rt_fll_set_freq+0x12>
1c00afbe:	c62e                	sw	a1,12(sp)
1c00afc0:	3595                	jal	1c00ae24 <__rt_bridge_req_shutdown>
1c00afc2:	45b2                	lw	a1,12(sp)
1c00afc4:	10059733          	p.fl1	a4,a1
1c00afc8:	47f5                	li	a5,29
1c00afca:	4505                	li	a0,1
1c00afcc:	82e7b7db          	p.subun	a5,a5,a4,1
1c00afd0:	04f567b3          	p.max	a5,a0,a5
1c00afd4:	fff78713          	addi	a4,a5,-1
1c00afd8:	00f595b3          	sll	a1,a1,a5
1c00afdc:	00e51533          	sll	a0,a0,a4
1c00afe0:	1c0016b7          	lui	a3,0x1c001
1c00afe4:	dc05b733          	p.bclr	a4,a1,14,0
1c00afe8:	c0f7255b          	p.addunr	a0,a4,a5
1c00afec:	7dc68693          	addi	a3,a3,2012 # 1c0017dc <__rt_fll_freq>
1c00aff0:	00241713          	slli	a4,s0,0x2
1c00aff4:	00a6e723          	p.sw	a0,a4(a3)
1c00aff8:	1c001737          	lui	a4,0x1c001
1c00affc:	74470713          	addi	a4,a4,1860 # 1c001744 <__rt_fll_is_on>
1c00b000:	9722                	add	a4,a4,s0
1c00b002:	00074703          	lbu	a4,0(a4)
1c00b006:	cf19                	beqz	a4,1c00b024 <__rt_fll_set_freq+0x72>
1c00b008:	0412                	slli	s0,s0,0x4
1c00b00a:	0411                	addi	s0,s0,4
1c00b00c:	1a1006b7          	lui	a3,0x1a100
1c00b010:	2086f703          	p.lw	a4,s0(a3)
1c00b014:	81bd                	srli	a1,a1,0xf
1c00b016:	de05a733          	p.insert	a4,a1,15,0
1c00b01a:	0785                	addi	a5,a5,1
1c00b01c:	c7a7a733          	p.insert	a4,a5,3,26
1c00b020:	00e6e423          	p.sw	a4,s0(a3)
1c00b024:	40f2                	lw	ra,28(sp)
1c00b026:	4462                	lw	s0,24(sp)
1c00b028:	6105                	addi	sp,sp,32
1c00b02a:	8082                	ret

1c00b02c <__rt_fll_init>:
1c00b02c:	1141                	addi	sp,sp,-16
1c00b02e:	00451613          	slli	a2,a0,0x4
1c00b032:	c226                	sw	s1,4(sp)
1c00b034:	00460813          	addi	a6,a2,4
1c00b038:	84aa                	mv	s1,a0
1c00b03a:	1a1006b7          	lui	a3,0x1a100
1c00b03e:	c606                	sw	ra,12(sp)
1c00b040:	c422                	sw	s0,8(sp)
1c00b042:	2106f703          	p.lw	a4,a6(a3)
1c00b046:	87ba                	mv	a5,a4
1c00b048:	04074163          	bltz	a4,1c00b08a <__rt_fll_init+0x5e>
1c00b04c:	00860893          	addi	a7,a2,8
1c00b050:	2116f503          	p.lw	a0,a7(a3)
1c00b054:	4599                	li	a1,6
1c00b056:	caa5a533          	p.insert	a0,a1,5,10
1c00b05a:	05000593          	li	a1,80
1c00b05e:	d705a533          	p.insert	a0,a1,11,16
1c00b062:	1a1005b7          	lui	a1,0x1a100
1c00b066:	00a5e8a3          	p.sw	a0,a7(a1)
1c00b06a:	0631                	addi	a2,a2,12
1c00b06c:	20c6f683          	p.lw	a3,a2(a3)
1c00b070:	14c00513          	li	a0,332
1c00b074:	d30526b3          	p.insert	a3,a0,9,16
1c00b078:	00d5e623          	p.sw	a3,a2(a1)
1c00b07c:	4685                	li	a3,1
1c00b07e:	c1e6a7b3          	p.insert	a5,a3,0,30
1c00b082:	c1f6a7b3          	p.insert	a5,a3,0,31
1c00b086:	00f5e823          	p.sw	a5,a6(a1)
1c00b08a:	1c001637          	lui	a2,0x1c001
1c00b08e:	00249693          	slli	a3,s1,0x2
1c00b092:	7dc60613          	addi	a2,a2,2012 # 1c0017dc <__rt_fll_freq>
1c00b096:	96b2                	add	a3,a3,a2
1c00b098:	4280                	lw	s0,0(a3)
1c00b09a:	c00d                	beqz	s0,1c00b0bc <__rt_fll_init+0x90>
1c00b09c:	85a2                	mv	a1,s0
1c00b09e:	8526                	mv	a0,s1
1c00b0a0:	3f09                	jal	1c00afb2 <__rt_fll_set_freq>
1c00b0a2:	1c0017b7          	lui	a5,0x1c001
1c00b0a6:	74478793          	addi	a5,a5,1860 # 1c001744 <__rt_fll_is_on>
1c00b0aa:	4705                	li	a4,1
1c00b0ac:	00e7c4a3          	p.sb	a4,s1(a5)
1c00b0b0:	8522                	mv	a0,s0
1c00b0b2:	40b2                	lw	ra,12(sp)
1c00b0b4:	4422                	lw	s0,8(sp)
1c00b0b6:	4492                	lw	s1,4(sp)
1c00b0b8:	0141                	addi	sp,sp,16
1c00b0ba:	8082                	ret
1c00b0bc:	10075733          	p.exthz	a4,a4
1c00b0c0:	c7a797b3          	p.extractu	a5,a5,3,26
1c00b0c4:	073e                	slli	a4,a4,0xf
1c00b0c6:	17fd                	addi	a5,a5,-1
1c00b0c8:	00f75433          	srl	s0,a4,a5
1c00b0cc:	c280                	sw	s0,0(a3)
1c00b0ce:	bfd1                	j	1c00b0a2 <__rt_fll_init+0x76>

1c00b0d0 <__rt_fll_deinit>:
1c00b0d0:	1c0017b7          	lui	a5,0x1c001
1c00b0d4:	74478793          	addi	a5,a5,1860 # 1c001744 <__rt_fll_is_on>
1c00b0d8:	0007c523          	p.sb	zero,a0(a5)
1c00b0dc:	8082                	ret

1c00b0de <__rt_flls_constructor>:
1c00b0de:	1c0017b7          	lui	a5,0x1c001
1c00b0e2:	7c07ae23          	sw	zero,2012(a5) # 1c0017dc <__rt_fll_freq>
1c00b0e6:	7dc78793          	addi	a5,a5,2012
1c00b0ea:	0007a223          	sw	zero,4(a5)
1c00b0ee:	0007a423          	sw	zero,8(a5)
1c00b0f2:	1c0017b7          	lui	a5,0x1c001
1c00b0f6:	74478793          	addi	a5,a5,1860 # 1c001744 <__rt_fll_is_on>
1c00b0fa:	00079023          	sh	zero,0(a5)
1c00b0fe:	00078123          	sb	zero,2(a5)
1c00b102:	8082                	ret

1c00b104 <rt_freq_set_and_get>:
1c00b104:	1101                	addi	sp,sp,-32
1c00b106:	cc22                	sw	s0,24(sp)
1c00b108:	c84a                	sw	s2,16(sp)
1c00b10a:	842a                	mv	s0,a0
1c00b10c:	892e                	mv	s2,a1
1c00b10e:	ce06                	sw	ra,28(sp)
1c00b110:	ca26                	sw	s1,20(sp)
1c00b112:	300474f3          	csrrci	s1,mstatus,8
1c00b116:	c632                	sw	a2,12(sp)
1c00b118:	3d69                	jal	1c00afb2 <__rt_fll_set_freq>
1c00b11a:	4632                	lw	a2,12(sp)
1c00b11c:	c211                	beqz	a2,1c00b120 <rt_freq_set_and_get+0x1c>
1c00b11e:	c208                	sw	a0,0(a2)
1c00b120:	1c0017b7          	lui	a5,0x1c001
1c00b124:	040a                	slli	s0,s0,0x2
1c00b126:	7e878793          	addi	a5,a5,2024 # 1c0017e8 <__rt_freq_domains>
1c00b12a:	0127e423          	p.sw	s2,s0(a5)
1c00b12e:	30049073          	csrw	mstatus,s1
1c00b132:	40f2                	lw	ra,28(sp)
1c00b134:	4462                	lw	s0,24(sp)
1c00b136:	44d2                	lw	s1,20(sp)
1c00b138:	4942                	lw	s2,16(sp)
1c00b13a:	4501                	li	a0,0
1c00b13c:	6105                	addi	sp,sp,32
1c00b13e:	8082                	ret

1c00b140 <__rt_freq_init>:
1c00b140:	1141                	addi	sp,sp,-16
1c00b142:	c606                	sw	ra,12(sp)
1c00b144:	c422                	sw	s0,8(sp)
1c00b146:	c226                	sw	s1,4(sp)
1c00b148:	3f59                	jal	1c00b0de <__rt_flls_constructor>
1c00b14a:	1c0014b7          	lui	s1,0x1c001
1c00b14e:	4505                	li	a0,1
1c00b150:	3df1                	jal	1c00b02c <__rt_fll_init>
1c00b152:	7e848413          	addi	s0,s1,2024 # 1c0017e8 <__rt_freq_domains>
1c00b156:	c048                	sw	a0,4(s0)
1c00b158:	4501                	li	a0,0
1c00b15a:	3dc9                	jal	1c00b02c <__rt_fll_init>
1c00b15c:	7ea4a423          	sw	a0,2024(s1)
1c00b160:	4509                	li	a0,2
1c00b162:	35e9                	jal	1c00b02c <__rt_fll_init>
1c00b164:	4795                	li	a5,5
1c00b166:	1a104737          	lui	a4,0x1a104
1c00b16a:	c408                	sw	a0,8(s0)
1c00b16c:	0cf72823          	sw	a5,208(a4) # 1a1040d0 <__l1_end+0xa1000b8>
1c00b170:	40b2                	lw	ra,12(sp)
1c00b172:	4422                	lw	s0,8(sp)
1c00b174:	4492                	lw	s1,4(sp)
1c00b176:	0141                	addi	sp,sp,16
1c00b178:	8082                	ret

1c00b17a <__rt_padframe_init>:
1c00b17a:	300477f3          	csrrci	a5,mstatus,8
1c00b17e:	30079073          	csrw	mstatus,a5
1c00b182:	8082                	ret

1c00b184 <rt_periph_copy>:
1c00b184:	7179                	addi	sp,sp,-48
1c00b186:	d422                	sw	s0,40(sp)
1c00b188:	842a                	mv	s0,a0
1c00b18a:	d606                	sw	ra,44(sp)
1c00b18c:	d226                	sw	s1,36(sp)
1c00b18e:	d04a                	sw	s2,32(sp)
1c00b190:	30047973          	csrrci	s2,mstatus,8
1c00b194:	4015d493          	srai	s1,a1,0x1
1c00b198:	1a102537          	lui	a0,0x1a102
1c00b19c:	08050513          	addi	a0,a0,128 # 1a102080 <__l1_end+0xa0fe068>
1c00b1a0:	049e                	slli	s1,s1,0x7
1c00b1a2:	94aa                	add	s1,s1,a0
1c00b1a4:	00459513          	slli	a0,a1,0x4
1c00b1a8:	8941                	andi	a0,a0,16
1c00b1aa:	94aa                	add	s1,s1,a0
1c00b1ac:	853e                	mv	a0,a5
1c00b1ae:	ef89                	bnez	a5,1c00b1c8 <rt_periph_copy+0x44>
1c00b1b0:	ce2e                	sw	a1,28(sp)
1c00b1b2:	cc32                	sw	a2,24(sp)
1c00b1b4:	ca36                	sw	a3,20(sp)
1c00b1b6:	c83a                	sw	a4,16(sp)
1c00b1b8:	c63e                	sw	a5,12(sp)
1c00b1ba:	c16fe0ef          	jal	ra,1c0095d0 <__rt_wait_event_prepare_blocking>
1c00b1be:	47b2                	lw	a5,12(sp)
1c00b1c0:	4742                	lw	a4,16(sp)
1c00b1c2:	46d2                	lw	a3,20(sp)
1c00b1c4:	4662                	lw	a2,24(sp)
1c00b1c6:	45f2                	lw	a1,28(sp)
1c00b1c8:	e419                	bnez	s0,1c00b1d6 <rt_periph_copy+0x52>
1c00b1ca:	03450413          	addi	s0,a0,52
1c00b1ce:	04052023          	sw	zero,64(a0)
1c00b1d2:	04052823          	sw	zero,80(a0)
1c00b1d6:	00c42803          	lw	a6,12(s0)
1c00b1da:	c054                	sw	a3,4(s0)
1c00b1dc:	cc08                	sw	a0,24(s0)
1c00b1de:	f6483833          	p.bclr	a6,a6,27,4
1c00b1e2:	4891                	li	a7,4
1c00b1e4:	c0474733          	p.bset	a4,a4,0,4
1c00b1e8:	0908e063          	bltu	a7,a6,1c00b268 <rt_periph_copy+0xe4>
1c00b1ec:	04c00893          	li	a7,76
1c00b1f0:	0596                	slli	a1,a1,0x5
1c00b1f2:	98ae                	add	a7,a7,a1
1c00b1f4:	0008a303          	lw	t1,0(a7)
1c00b1f8:	00042a23          	sw	zero,20(s0)
1c00b1fc:	04c00813          	li	a6,76
1c00b200:	04031463          	bnez	t1,1c00b248 <rt_periph_copy+0xc4>
1c00b204:	0088a023          	sw	s0,0(a7)
1c00b208:	00b808b3          	add	a7,a6,a1
1c00b20c:	0088a303          	lw	t1,8(a7)
1c00b210:	0088a223          	sw	s0,4(a7)
1c00b214:	02031f63          	bnez	t1,1c00b252 <rt_periph_copy+0xce>
1c00b218:	00848e13          	addi	t3,s1,8
1c00b21c:	000e2883          	lw	a7,0(t3)
1c00b220:	0208f893          	andi	a7,a7,32
1c00b224:	02089763          	bnez	a7,1c00b252 <rt_periph_copy+0xce>
1c00b228:	00c4a22b          	p.sw	a2,4(s1!)
1c00b22c:	c094                	sw	a3,0(s1)
1c00b22e:	00ee2023          	sw	a4,0(t3)
1c00b232:	e399                	bnez	a5,1c00b238 <rt_periph_copy+0xb4>
1c00b234:	ceafe0ef          	jal	ra,1c00971e <__rt_wait_event>
1c00b238:	30091073          	csrw	mstatus,s2
1c00b23c:	50b2                	lw	ra,44(sp)
1c00b23e:	5422                	lw	s0,40(sp)
1c00b240:	5492                	lw	s1,36(sp)
1c00b242:	5902                	lw	s2,32(sp)
1c00b244:	6145                	addi	sp,sp,48
1c00b246:	8082                	ret
1c00b248:	0048a883          	lw	a7,4(a7)
1c00b24c:	0088aa23          	sw	s0,20(a7)
1c00b250:	bf65                	j	1c00b208 <rt_periph_copy+0x84>
1c00b252:	00042823          	sw	zero,16(s0)
1c00b256:	c010                	sw	a2,0(s0)
1c00b258:	c054                	sw	a3,4(s0)
1c00b25a:	c418                	sw	a4,8(s0)
1c00b25c:	fc031be3          	bnez	t1,1c00b232 <rt_periph_copy+0xae>
1c00b260:	982e                	add	a6,a6,a1
1c00b262:	00882423          	sw	s0,8(a6)
1c00b266:	b7f1                	j	1c00b232 <rt_periph_copy+0xae>
1c00b268:	fc6835e3          	p.bneimm	a6,6,1c00b232 <rt_periph_copy+0xae>
1c00b26c:	04c00893          	li	a7,76
1c00b270:	0596                	slli	a1,a1,0x5
1c00b272:	98ae                	add	a7,a7,a1
1c00b274:	0008a303          	lw	t1,0(a7)
1c00b278:	00042a23          	sw	zero,20(s0)
1c00b27c:	04c00813          	li	a6,76
1c00b280:	02031563          	bnez	t1,1c00b2aa <rt_periph_copy+0x126>
1c00b284:	0088a023          	sw	s0,0(a7)
1c00b288:	95c2                	add	a1,a1,a6
1c00b28a:	c1c0                	sw	s0,4(a1)
1c00b28c:	02031463          	bnez	t1,1c00b2b4 <rt_periph_copy+0x130>
1c00b290:	02442803          	lw	a6,36(s0)
1c00b294:	1a1025b7          	lui	a1,0x1a102
1c00b298:	4b05a023          	sw	a6,1184(a1) # 1a1024a0 <__l1_end+0xa0fe488>
1c00b29c:	85a6                	mv	a1,s1
1c00b29e:	00c5a22b          	p.sw	a2,4(a1!)
1c00b2a2:	c194                	sw	a3,0(a1)
1c00b2a4:	04a1                	addi	s1,s1,8
1c00b2a6:	c098                	sw	a4,0(s1)
1c00b2a8:	b769                	j	1c00b232 <rt_periph_copy+0xae>
1c00b2aa:	0048a883          	lw	a7,4(a7)
1c00b2ae:	0088aa23          	sw	s0,20(a7)
1c00b2b2:	bfd9                	j	1c00b288 <rt_periph_copy+0x104>
1c00b2b4:	c418                	sw	a4,8(s0)
1c00b2b6:	4598                	lw	a4,8(a1)
1c00b2b8:	c010                	sw	a2,0(s0)
1c00b2ba:	c054                	sw	a3,4(s0)
1c00b2bc:	00042823          	sw	zero,16(s0)
1c00b2c0:	fb2d                	bnez	a4,1c00b232 <rt_periph_copy+0xae>
1c00b2c2:	c580                	sw	s0,8(a1)
1c00b2c4:	b7bd                	j	1c00b232 <rt_periph_copy+0xae>

1c00b2c6 <__rt_periph_init>:
1c00b2c6:	04c00693          	li	a3,76
1c00b2ca:	1c009637          	lui	a2,0x1c009
1c00b2ce:	42068693          	addi	a3,a3,1056 # 1a100420 <__l1_end+0xa0fc408>
1c00b2d2:	04c00713          	li	a4,76
1c00b2d6:	ec260613          	addi	a2,a2,-318 # 1c008ec2 <udma_event_handler>
1c00b2da:	010250fb          	lp.setupi	x1,16,1c00b2e2 <__rt_periph_init+0x1c>
1c00b2de:	00c6a22b          	p.sw	a2,4(a3!)
1c00b2e2:	0001                	nop
1c00b2e4:	40072023          	sw	zero,1024(a4)
1c00b2e8:	40072223          	sw	zero,1028(a4)
1c00b2ec:	40072423          	sw	zero,1032(a4)
1c00b2f0:	40072623          	sw	zero,1036(a4)
1c00b2f4:	40072823          	sw	zero,1040(a4)
1c00b2f8:	40072a23          	sw	zero,1044(a4)
1c00b2fc:	40072c23          	sw	zero,1048(a4)
1c00b300:	1a102837          	lui	a6,0x1a102
1c00b304:	40072e23          	sw	zero,1052(a4)
1c00b308:	04c00793          	li	a5,76
1c00b30c:	4681                	li	a3,0
1c00b30e:	08080813          	addi	a6,a6,128 # 1a102080 <__l1_end+0xa0fe068>
1c00b312:	0208d0fb          	lp.setupi	x1,32,1c00b334 <__rt_periph_init+0x6e>
1c00b316:	4016d713          	srai	a4,a3,0x1
1c00b31a:	00469513          	slli	a0,a3,0x4
1c00b31e:	071e                	slli	a4,a4,0x7
1c00b320:	9742                	add	a4,a4,a6
1c00b322:	8941                	andi	a0,a0,16
1c00b324:	972a                	add	a4,a4,a0
1c00b326:	0007a023          	sw	zero,0(a5)
1c00b32a:	0007a423          	sw	zero,8(a5)
1c00b32e:	c7d8                	sw	a4,12(a5)
1c00b330:	cfd0                	sw	a2,28(a5)
1c00b332:	0685                	addi	a3,a3,1
1c00b334:	02078793          	addi	a5,a5,32
1c00b338:	8082                	ret

1c00b33a <__rt_i2c_init>:
1c00b33a:	1c0107b7          	lui	a5,0x1c010
1c00b33e:	18078223          	sb	zero,388(a5) # 1c010184 <__cluster_text_end+0x4>
1c00b342:	8082                	ret

1c00b344 <__rt_rtc_init>:
1c00b344:	4ec00793          	li	a5,1260
1c00b348:	0207ac23          	sw	zero,56(a5)
1c00b34c:	02078823          	sb	zero,48(a5)
1c00b350:	0207aa23          	sw	zero,52(a5)
1c00b354:	8082                	ret

1c00b356 <__rt_hyper_init>:
1c00b356:	1c001737          	lui	a4,0x1c001
1c00b35a:	52800793          	li	a5,1320
1c00b35e:	74072423          	sw	zero,1864(a4) # 1c001748 <__pi_hyper_cluster_reqs_first>
1c00b362:	577d                	li	a4,-1
1c00b364:	0007aa23          	sw	zero,20(a5)
1c00b368:	0207a823          	sw	zero,48(a5)
1c00b36c:	cf98                	sw	a4,24(a5)
1c00b36e:	8082                	ret

1c00b370 <rt_is_fc>:
1c00b370:	f1402573          	csrr	a0,mhartid
1c00b374:	8515                	srai	a0,a0,0x5
1c00b376:	f2653533          	p.bclr	a0,a0,25,6
1c00b37a:	1505                	addi	a0,a0,-31
1c00b37c:	00153513          	seqz	a0,a0
1c00b380:	8082                	ret

1c00b382 <__rt_io_end_of_flush>:
1c00b382:	1c0017b7          	lui	a5,0x1c001
1c00b386:	7407a823          	sw	zero,1872(a5) # 1c001750 <__rt_io_pending_flush>
1c00b38a:	00052c23          	sw	zero,24(a0)
1c00b38e:	8082                	ret

1c00b390 <__rt_io_uart_wait_req>:
1c00b390:	1141                	addi	sp,sp,-16
1c00b392:	c226                	sw	s1,4(sp)
1c00b394:	84aa                	mv	s1,a0
1c00b396:	c606                	sw	ra,12(sp)
1c00b398:	c422                	sw	s0,8(sp)
1c00b39a:	c04a                	sw	s2,0(sp)
1c00b39c:	30047973          	csrrci	s2,mstatus,8
1c00b3a0:	1c001437          	lui	s0,0x1c001
1c00b3a4:	74c40413          	addi	s0,s0,1868 # 1c00174c <__rt_io_event_current>
1c00b3a8:	4008                	lw	a0,0(s0)
1c00b3aa:	c509                	beqz	a0,1c00b3b4 <__rt_io_uart_wait_req+0x24>
1c00b3ac:	b9cfe0ef          	jal	ra,1c009748 <rt_event_wait>
1c00b3b0:	00042023          	sw	zero,0(s0)
1c00b3b4:	4785                	li	a5,1
1c00b3b6:	08f48623          	sb	a5,140(s1)
1c00b3ba:	08d4c783          	lbu	a5,141(s1)
1c00b3be:	00201737          	lui	a4,0x201
1c00b3c2:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e4e1c>
1c00b3c6:	04078793          	addi	a5,a5,64
1c00b3ca:	07da                	slli	a5,a5,0x16
1c00b3cc:	0007e723          	p.sw	zero,a4(a5)
1c00b3d0:	30091073          	csrw	mstatus,s2
1c00b3d4:	40b2                	lw	ra,12(sp)
1c00b3d6:	4422                	lw	s0,8(sp)
1c00b3d8:	4492                	lw	s1,4(sp)
1c00b3da:	4902                	lw	s2,0(sp)
1c00b3dc:	0141                	addi	sp,sp,16
1c00b3de:	8082                	ret

1c00b3e0 <__rt_io_start>:
1c00b3e0:	1101                	addi	sp,sp,-32
1c00b3e2:	0028                	addi	a0,sp,8
1c00b3e4:	ce06                	sw	ra,28(sp)
1c00b3e6:	cc22                	sw	s0,24(sp)
1c00b3e8:	006010ef          	jal	ra,1c00c3ee <rt_uart_conf_init>
1c00b3ec:	4585                	li	a1,1
1c00b3ee:	4501                	li	a0,0
1c00b3f0:	9f8fe0ef          	jal	ra,1c0095e8 <rt_event_alloc>
1c00b3f4:	547d                	li	s0,-1
1c00b3f6:	ed1d                	bnez	a0,1c00b434 <__rt_io_start+0x54>
1c00b3f8:	1c0017b7          	lui	a5,0x1c001
1c00b3fc:	6487a783          	lw	a5,1608(a5) # 1c001648 <__rt_iodev_uart_baudrate>
1c00b400:	842a                	mv	s0,a0
1c00b402:	1c001537          	lui	a0,0x1c001
1c00b406:	e3ff5597          	auipc	a1,0xe3ff5
1c00b40a:	c0658593          	addi	a1,a1,-1018 # c <__rt_sched>
1c00b40e:	67c50513          	addi	a0,a0,1660 # 1c00167c <__rt_io_event>
1c00b412:	c43e                	sw	a5,8(sp)
1c00b414:	9aafe0ef          	jal	ra,1c0095be <__rt_event_init>
1c00b418:	1c0017b7          	lui	a5,0x1c001
1c00b41c:	75c7a503          	lw	a0,1884(a5) # 1c00175c <__rt_iodev_uart_channel>
1c00b420:	4681                	li	a3,0
1c00b422:	4601                	li	a2,0
1c00b424:	002c                	addi	a1,sp,8
1c00b426:	0511                	addi	a0,a0,4
1c00b428:	7d7000ef          	jal	ra,1c00c3fe <__rt_uart_open>
1c00b42c:	1c0017b7          	lui	a5,0x1c001
1c00b430:	74a7aa23          	sw	a0,1876(a5) # 1c001754 <_rt_io_uart>
1c00b434:	8522                	mv	a0,s0
1c00b436:	40f2                	lw	ra,28(sp)
1c00b438:	4462                	lw	s0,24(sp)
1c00b43a:	6105                	addi	sp,sp,32
1c00b43c:	8082                	ret

1c00b43e <rt_event_execute.isra.2.constprop.11>:
1c00b43e:	1141                	addi	sp,sp,-16
1c00b440:	c606                	sw	ra,12(sp)
1c00b442:	c422                	sw	s0,8(sp)
1c00b444:	30047473          	csrrci	s0,mstatus,8
1c00b448:	4585                	li	a1,1
1c00b44a:	e3ff5517          	auipc	a0,0xe3ff5
1c00b44e:	bc250513          	addi	a0,a0,-1086 # c <__rt_sched>
1c00b452:	a6cfe0ef          	jal	ra,1c0096be <__rt_event_execute>
1c00b456:	30041073          	csrw	mstatus,s0
1c00b45a:	40b2                	lw	ra,12(sp)
1c00b45c:	4422                	lw	s0,8(sp)
1c00b45e:	0141                	addi	sp,sp,16
1c00b460:	8082                	ret

1c00b462 <__rt_io_lock>:
1c00b462:	1c0017b7          	lui	a5,0x1c001
1c00b466:	5947a783          	lw	a5,1428(a5) # 1c001594 <__hal_debug_struct+0x10>
1c00b46a:	c791                	beqz	a5,1c00b476 <__rt_io_lock+0x14>
1c00b46c:	1c0017b7          	lui	a5,0x1c001
1c00b470:	7547a783          	lw	a5,1876(a5) # 1c001754 <_rt_io_uart>
1c00b474:	c3a1                	beqz	a5,1c00b4b4 <__rt_io_lock+0x52>
1c00b476:	7171                	addi	sp,sp,-176
1c00b478:	d706                	sw	ra,172(sp)
1c00b47a:	3ddd                	jal	1c00b370 <rt_is_fc>
1c00b47c:	1c0017b7          	lui	a5,0x1c001
1c00b480:	c901                	beqz	a0,1c00b490 <__rt_io_lock+0x2e>
1c00b482:	57478513          	addi	a0,a5,1396 # 1c001574 <__rt_io_fc_lock>
1c00b486:	eecff0ef          	jal	ra,1c00ab72 <__rt_fc_lock>
1c00b48a:	50ba                	lw	ra,172(sp)
1c00b48c:	614d                	addi	sp,sp,176
1c00b48e:	8082                	ret
1c00b490:	002c                	addi	a1,sp,8
1c00b492:	57478513          	addi	a0,a5,1396
1c00b496:	f4eff0ef          	jal	ra,1c00abe4 <__rt_fc_cluster_lock>
1c00b49a:	4689                	li	a3,2
1c00b49c:	00204737          	lui	a4,0x204
1c00b4a0:	09c14783          	lbu	a5,156(sp)
1c00b4a4:	0ff7f793          	andi	a5,a5,255
1c00b4a8:	f3ed                	bnez	a5,1c00b48a <__rt_io_lock+0x28>
1c00b4aa:	c714                	sw	a3,8(a4)
1c00b4ac:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1e8054>
1c00b4b0:	c354                	sw	a3,4(a4)
1c00b4b2:	b7fd                	j	1c00b4a0 <__rt_io_lock+0x3e>
1c00b4b4:	8082                	ret

1c00b4b6 <__rt_io_unlock>:
1c00b4b6:	1c0017b7          	lui	a5,0x1c001
1c00b4ba:	5947a783          	lw	a5,1428(a5) # 1c001594 <__hal_debug_struct+0x10>
1c00b4be:	c791                	beqz	a5,1c00b4ca <__rt_io_unlock+0x14>
1c00b4c0:	1c0017b7          	lui	a5,0x1c001
1c00b4c4:	7547a783          	lw	a5,1876(a5) # 1c001754 <_rt_io_uart>
1c00b4c8:	c3a1                	beqz	a5,1c00b508 <__rt_io_unlock+0x52>
1c00b4ca:	7171                	addi	sp,sp,-176
1c00b4cc:	d706                	sw	ra,172(sp)
1c00b4ce:	354d                	jal	1c00b370 <rt_is_fc>
1c00b4d0:	1c0017b7          	lui	a5,0x1c001
1c00b4d4:	c901                	beqz	a0,1c00b4e4 <__rt_io_unlock+0x2e>
1c00b4d6:	57478513          	addi	a0,a5,1396 # 1c001574 <__rt_io_fc_lock>
1c00b4da:	ed6ff0ef          	jal	ra,1c00abb0 <__rt_fc_unlock>
1c00b4de:	50ba                	lw	ra,172(sp)
1c00b4e0:	614d                	addi	sp,sp,176
1c00b4e2:	8082                	ret
1c00b4e4:	002c                	addi	a1,sp,8
1c00b4e6:	57478513          	addi	a0,a5,1396
1c00b4ea:	f32ff0ef          	jal	ra,1c00ac1c <__rt_fc_cluster_unlock>
1c00b4ee:	4689                	li	a3,2
1c00b4f0:	00204737          	lui	a4,0x204
1c00b4f4:	09c14783          	lbu	a5,156(sp)
1c00b4f8:	0ff7f793          	andi	a5,a5,255
1c00b4fc:	f3ed                	bnez	a5,1c00b4de <__rt_io_unlock+0x28>
1c00b4fe:	c714                	sw	a3,8(a4)
1c00b500:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1e8054>
1c00b504:	c354                	sw	a3,4(a4)
1c00b506:	b7fd                	j	1c00b4f4 <__rt_io_unlock+0x3e>
1c00b508:	8082                	ret

1c00b50a <__rt_io_uart_wait_pending>:
1c00b50a:	7135                	addi	sp,sp,-160
1c00b50c:	cd22                	sw	s0,152(sp)
1c00b50e:	cf06                	sw	ra,156(sp)
1c00b510:	cb26                	sw	s1,148(sp)
1c00b512:	1c001437          	lui	s0,0x1c001
1c00b516:	75042783          	lw	a5,1872(s0) # 1c001750 <__rt_io_pending_flush>
1c00b51a:	e39d                	bnez	a5,1c00b540 <__rt_io_uart_wait_pending+0x36>
1c00b51c:	1c001437          	lui	s0,0x1c001
1c00b520:	74c40413          	addi	s0,s0,1868 # 1c00174c <__rt_io_event_current>
1c00b524:	4004                	lw	s1,0(s0)
1c00b526:	c881                	beqz	s1,1c00b536 <__rt_io_uart_wait_pending+0x2c>
1c00b528:	35a1                	jal	1c00b370 <rt_is_fc>
1c00b52a:	cd19                	beqz	a0,1c00b548 <__rt_io_uart_wait_pending+0x3e>
1c00b52c:	8526                	mv	a0,s1
1c00b52e:	a1afe0ef          	jal	ra,1c009748 <rt_event_wait>
1c00b532:	00042023          	sw	zero,0(s0)
1c00b536:	40fa                	lw	ra,156(sp)
1c00b538:	446a                	lw	s0,152(sp)
1c00b53a:	44da                	lw	s1,148(sp)
1c00b53c:	610d                	addi	sp,sp,160
1c00b53e:	8082                	ret
1c00b540:	3f9d                	jal	1c00b4b6 <__rt_io_unlock>
1c00b542:	3df5                	jal	1c00b43e <rt_event_execute.isra.2.constprop.11>
1c00b544:	3f39                	jal	1c00b462 <__rt_io_lock>
1c00b546:	bfc1                	j	1c00b516 <__rt_io_uart_wait_pending+0xc>
1c00b548:	f14027f3          	csrr	a5,mhartid
1c00b54c:	8795                	srai	a5,a5,0x5
1c00b54e:	f267b7b3          	p.bclr	a5,a5,25,6
1c00b552:	08f106a3          	sb	a5,141(sp)
1c00b556:	1c00b7b7          	lui	a5,0x1c00b
1c00b55a:	39078793          	addi	a5,a5,912 # 1c00b390 <__rt_io_uart_wait_req>
1c00b55e:	c03e                	sw	a5,0(sp)
1c00b560:	00010793          	mv	a5,sp
1c00b564:	4705                	li	a4,1
1c00b566:	c23e                	sw	a5,4(sp)
1c00b568:	850a                	mv	a0,sp
1c00b56a:	1c0017b7          	lui	a5,0x1c001
1c00b56e:	68e7ae23          	sw	a4,1692(a5) # 1c00169c <__rt_io_event+0x20>
1c00b572:	08010623          	sb	zero,140(sp)
1c00b576:	d002                	sw	zero,32(sp)
1c00b578:	d202                	sw	zero,36(sp)
1c00b57a:	d35fe0ef          	jal	ra,1c00a2ae <__rt_cluster_push_fc_event>
1c00b57e:	4689                	li	a3,2
1c00b580:	00204737          	lui	a4,0x204
1c00b584:	08c14783          	lbu	a5,140(sp)
1c00b588:	0ff7f793          	andi	a5,a5,255
1c00b58c:	f7cd                	bnez	a5,1c00b536 <__rt_io_uart_wait_pending+0x2c>
1c00b58e:	c714                	sw	a3,8(a4)
1c00b590:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1e8054>
1c00b594:	c354                	sw	a3,4(a4)
1c00b596:	b7fd                	j	1c00b584 <__rt_io_uart_wait_pending+0x7a>

1c00b598 <__rt_io_stop>:
1c00b598:	1141                	addi	sp,sp,-16
1c00b59a:	c422                	sw	s0,8(sp)
1c00b59c:	1c001437          	lui	s0,0x1c001
1c00b5a0:	c606                	sw	ra,12(sp)
1c00b5a2:	75440413          	addi	s0,s0,1876 # 1c001754 <_rt_io_uart>
1c00b5a6:	3795                	jal	1c00b50a <__rt_io_uart_wait_pending>
1c00b5a8:	4008                	lw	a0,0(s0)
1c00b5aa:	4581                	li	a1,0
1c00b5ac:	6d7000ef          	jal	ra,1c00c482 <rt_uart_close>
1c00b5b0:	40b2                	lw	ra,12(sp)
1c00b5b2:	00042023          	sw	zero,0(s0)
1c00b5b6:	4422                	lw	s0,8(sp)
1c00b5b8:	4501                	li	a0,0
1c00b5ba:	0141                	addi	sp,sp,16
1c00b5bc:	8082                	ret

1c00b5be <__rt_io_uart_flush.constprop.10>:
1c00b5be:	7131                	addi	sp,sp,-192
1c00b5c0:	dd22                	sw	s0,184(sp)
1c00b5c2:	df06                	sw	ra,188(sp)
1c00b5c4:	db26                	sw	s1,180(sp)
1c00b5c6:	d94a                	sw	s2,176(sp)
1c00b5c8:	d74e                	sw	s3,172(sp)
1c00b5ca:	d552                	sw	s4,168(sp)
1c00b5cc:	d356                	sw	s5,164(sp)
1c00b5ce:	1c001437          	lui	s0,0x1c001
1c00b5d2:	75042783          	lw	a5,1872(s0) # 1c001750 <__rt_io_pending_flush>
1c00b5d6:	75040a13          	addi	s4,s0,1872
1c00b5da:	e7bd                	bnez	a5,1c00b648 <__rt_io_uart_flush.constprop.10+0x8a>
1c00b5dc:	1c0014b7          	lui	s1,0x1c001
1c00b5e0:	58448793          	addi	a5,s1,1412 # 1c001584 <__hal_debug_struct>
1c00b5e4:	4f80                	lw	s0,24(a5)
1c00b5e6:	58448a93          	addi	s5,s1,1412
1c00b5ea:	c431                	beqz	s0,1c00b636 <__rt_io_uart_flush.constprop.10+0x78>
1c00b5ec:	3351                	jal	1c00b370 <rt_is_fc>
1c00b5ee:	1c0017b7          	lui	a5,0x1c001
1c00b5f2:	7547a903          	lw	s2,1876(a5) # 1c001754 <_rt_io_uart>
1c00b5f6:	1c0019b7          	lui	s3,0x1c001
1c00b5fa:	cd29                	beqz	a0,1c00b654 <__rt_io_uart_flush.constprop.10+0x96>
1c00b5fc:	1c00b5b7          	lui	a1,0x1c00b
1c00b600:	4785                	li	a5,1
1c00b602:	58448613          	addi	a2,s1,1412
1c00b606:	38258593          	addi	a1,a1,898 # 1c00b382 <__rt_io_end_of_flush>
1c00b60a:	4501                	li	a0,0
1c00b60c:	00fa2023          	sw	a5,0(s4)
1c00b610:	84efe0ef          	jal	ra,1c00965e <rt_event_get>
1c00b614:	00492583          	lw	a1,4(s2)
1c00b618:	87aa                	mv	a5,a0
1c00b61a:	4701                	li	a4,0
1c00b61c:	0586                	slli	a1,a1,0x1
1c00b61e:	86a2                	mv	a3,s0
1c00b620:	5a098613          	addi	a2,s3,1440 # 1c0015a0 <__hal_debug_struct+0x1c>
1c00b624:	0585                	addi	a1,a1,1
1c00b626:	4501                	li	a0,0
1c00b628:	b5dff0ef          	jal	ra,1c00b184 <rt_periph_copy>
1c00b62c:	3569                	jal	1c00b4b6 <__rt_io_unlock>
1c00b62e:	000a2783          	lw	a5,0(s4)
1c00b632:	ef99                	bnez	a5,1c00b650 <__rt_io_uart_flush.constprop.10+0x92>
1c00b634:	353d                	jal	1c00b462 <__rt_io_lock>
1c00b636:	50fa                	lw	ra,188(sp)
1c00b638:	546a                	lw	s0,184(sp)
1c00b63a:	54da                	lw	s1,180(sp)
1c00b63c:	594a                	lw	s2,176(sp)
1c00b63e:	59ba                	lw	s3,172(sp)
1c00b640:	5a2a                	lw	s4,168(sp)
1c00b642:	5a9a                	lw	s5,164(sp)
1c00b644:	6129                	addi	sp,sp,192
1c00b646:	8082                	ret
1c00b648:	35bd                	jal	1c00b4b6 <__rt_io_unlock>
1c00b64a:	3bd5                	jal	1c00b43e <rt_event_execute.isra.2.constprop.11>
1c00b64c:	3d19                	jal	1c00b462 <__rt_io_lock>
1c00b64e:	b751                	j	1c00b5d2 <__rt_io_uart_flush.constprop.10+0x14>
1c00b650:	33fd                	jal	1c00b43e <rt_event_execute.isra.2.constprop.11>
1c00b652:	bff1                	j	1c00b62e <__rt_io_uart_flush.constprop.10+0x70>
1c00b654:	0054                	addi	a3,sp,4
1c00b656:	8622                	mv	a2,s0
1c00b658:	5a098593          	addi	a1,s3,1440
1c00b65c:	854a                	mv	a0,s2
1c00b65e:	675000ef          	jal	ra,1c00c4d2 <rt_uart_cluster_write>
1c00b662:	4689                	li	a3,2
1c00b664:	00204737          	lui	a4,0x204
1c00b668:	09c14783          	lbu	a5,156(sp)
1c00b66c:	0ff7f793          	andi	a5,a5,255
1c00b670:	c781                	beqz	a5,1c00b678 <__rt_io_uart_flush.constprop.10+0xba>
1c00b672:	000aac23          	sw	zero,24(s5)
1c00b676:	b7c1                	j	1c00b636 <__rt_io_uart_flush.constprop.10+0x78>
1c00b678:	c714                	sw	a3,8(a4)
1c00b67a:	03c76783          	p.elw	a5,60(a4) # 20403c <__l1_heap_size+0x1e8054>
1c00b67e:	c354                	sw	a3,4(a4)
1c00b680:	b7e5                	j	1c00b668 <__rt_io_uart_flush.constprop.10+0xaa>

1c00b682 <memset>:
1c00b682:	962a                	add	a2,a2,a0
1c00b684:	87aa                	mv	a5,a0
1c00b686:	00c79363          	bne	a5,a2,1c00b68c <memset+0xa>
1c00b68a:	8082                	ret
1c00b68c:	00b780ab          	p.sb	a1,1(a5!)
1c00b690:	bfdd                	j	1c00b686 <memset+0x4>

1c00b692 <memcpy>:
1c00b692:	00a5e733          	or	a4,a1,a0
1c00b696:	fa273733          	p.bclr	a4,a4,29,2
1c00b69a:	87aa                	mv	a5,a0
1c00b69c:	c709                	beqz	a4,1c00b6a6 <memcpy+0x14>
1c00b69e:	962e                	add	a2,a2,a1
1c00b6a0:	00c59f63          	bne	a1,a2,1c00b6be <memcpy+0x2c>
1c00b6a4:	8082                	ret
1c00b6a6:	fa263733          	p.bclr	a4,a2,29,2
1c00b6aa:	fb75                	bnez	a4,1c00b69e <memcpy+0xc>
1c00b6ac:	962e                	add	a2,a2,a1
1c00b6ae:	00c59363          	bne	a1,a2,1c00b6b4 <memcpy+0x22>
1c00b6b2:	8082                	ret
1c00b6b4:	0045a70b          	p.lw	a4,4(a1!)
1c00b6b8:	00e7a22b          	p.sw	a4,4(a5!)
1c00b6bc:	bfcd                	j	1c00b6ae <memcpy+0x1c>
1c00b6be:	0015c70b          	p.lbu	a4,1(a1!)
1c00b6c2:	00e780ab          	p.sb	a4,1(a5!)
1c00b6c6:	bfe9                	j	1c00b6a0 <memcpy+0xe>

1c00b6c8 <memmove>:
1c00b6c8:	40b507b3          	sub	a5,a0,a1
1c00b6cc:	00c7e763          	bltu	a5,a2,1c00b6da <memmove+0x12>
1c00b6d0:	962a                	add	a2,a2,a0
1c00b6d2:	87aa                	mv	a5,a0
1c00b6d4:	00c79f63          	bne	a5,a2,1c00b6f2 <memmove+0x2a>
1c00b6d8:	8082                	ret
1c00b6da:	167d                	addi	a2,a2,-1
1c00b6dc:	00c507b3          	add	a5,a0,a2
1c00b6e0:	95b2                	add	a1,a1,a2
1c00b6e2:	0605                	addi	a2,a2,1
1c00b6e4:	004640fb          	lp.setup	x1,a2,1c00b6ec <memmove+0x24>
1c00b6e8:	fff5c70b          	p.lbu	a4,-1(a1!)
1c00b6ec:	fee78fab          	p.sb	a4,-1(a5!)
1c00b6f0:	8082                	ret
1c00b6f2:	0015c70b          	p.lbu	a4,1(a1!)
1c00b6f6:	00e780ab          	p.sb	a4,1(a5!)
1c00b6fa:	bfe9                	j	1c00b6d4 <memmove+0xc>

1c00b6fc <strchr>:
1c00b6fc:	0ff5f593          	andi	a1,a1,255
1c00b700:	00054703          	lbu	a4,0(a0)
1c00b704:	87aa                	mv	a5,a0
1c00b706:	0505                	addi	a0,a0,1
1c00b708:	00b70563          	beq	a4,a1,1c00b712 <strchr+0x16>
1c00b70c:	fb75                	bnez	a4,1c00b700 <strchr+0x4>
1c00b70e:	c191                	beqz	a1,1c00b712 <strchr+0x16>
1c00b710:	4781                	li	a5,0
1c00b712:	853e                	mv	a0,a5
1c00b714:	8082                	ret

1c00b716 <__rt_putc_debug_bridge>:
1c00b716:	1141                	addi	sp,sp,-16
1c00b718:	c422                	sw	s0,8(sp)
1c00b71a:	1c001437          	lui	s0,0x1c001
1c00b71e:	c226                	sw	s1,4(sp)
1c00b720:	c606                	sw	ra,12(sp)
1c00b722:	84aa                	mv	s1,a0
1c00b724:	58440413          	addi	s0,s0,1412 # 1c001584 <__hal_debug_struct>
1c00b728:	485c                	lw	a5,20(s0)
1c00b72a:	c791                	beqz	a5,1c00b736 <__rt_putc_debug_bridge+0x20>
1c00b72c:	06400513          	li	a0,100
1c00b730:	ca6fe0ef          	jal	ra,1c009bd6 <rt_time_wait_us>
1c00b734:	bfd5                	j	1c00b728 <__rt_putc_debug_bridge+0x12>
1c00b736:	4c1c                	lw	a5,24(s0)
1c00b738:	00178713          	addi	a4,a5,1
1c00b73c:	97a2                	add	a5,a5,s0
1c00b73e:	00978e23          	sb	s1,28(a5)
1c00b742:	cc18                	sw	a4,24(s0)
1c00b744:	4c14                	lw	a3,24(s0)
1c00b746:	08000793          	li	a5,128
1c00b74a:	00f68463          	beq	a3,a5,1c00b752 <__rt_putc_debug_bridge+0x3c>
1c00b74e:	00a4b663          	p.bneimm	s1,10,1c00b75a <__rt_putc_debug_bridge+0x44>
1c00b752:	c701                	beqz	a4,1c00b75a <__rt_putc_debug_bridge+0x44>
1c00b754:	c858                	sw	a4,20(s0)
1c00b756:	00042c23          	sw	zero,24(s0)
1c00b75a:	4c1c                	lw	a5,24(s0)
1c00b75c:	e799                	bnez	a5,1c00b76a <__rt_putc_debug_bridge+0x54>
1c00b75e:	4422                	lw	s0,8(sp)
1c00b760:	40b2                	lw	ra,12(sp)
1c00b762:	4492                	lw	s1,4(sp)
1c00b764:	0141                	addi	sp,sp,16
1c00b766:	e84ff06f          	j	1c00adea <__rt_bridge_printf_flush>
1c00b76a:	40b2                	lw	ra,12(sp)
1c00b76c:	4422                	lw	s0,8(sp)
1c00b76e:	4492                	lw	s1,4(sp)
1c00b770:	0141                	addi	sp,sp,16
1c00b772:	8082                	ret

1c00b774 <__rt_putc_uart>:
1c00b774:	1101                	addi	sp,sp,-32
1c00b776:	c62a                	sw	a0,12(sp)
1c00b778:	ce06                	sw	ra,28(sp)
1c00b77a:	3b41                	jal	1c00b50a <__rt_io_uart_wait_pending>
1c00b77c:	1c0017b7          	lui	a5,0x1c001
1c00b780:	58478793          	addi	a5,a5,1412 # 1c001584 <__hal_debug_struct>
1c00b784:	4f94                	lw	a3,24(a5)
1c00b786:	4532                	lw	a0,12(sp)
1c00b788:	00168713          	addi	a4,a3,1
1c00b78c:	cf98                	sw	a4,24(a5)
1c00b78e:	97b6                	add	a5,a5,a3
1c00b790:	00a78e23          	sb	a0,28(a5)
1c00b794:	08000793          	li	a5,128
1c00b798:	00f70463          	beq	a4,a5,1c00b7a0 <__rt_putc_uart+0x2c>
1c00b79c:	00a53563          	p.bneimm	a0,10,1c00b7a6 <__rt_putc_uart+0x32>
1c00b7a0:	40f2                	lw	ra,28(sp)
1c00b7a2:	6105                	addi	sp,sp,32
1c00b7a4:	bd29                	j	1c00b5be <__rt_io_uart_flush.constprop.10>
1c00b7a6:	40f2                	lw	ra,28(sp)
1c00b7a8:	6105                	addi	sp,sp,32
1c00b7aa:	8082                	ret

1c00b7ac <tfp_putc.isra.8>:
1c00b7ac:	1c0017b7          	lui	a5,0x1c001
1c00b7b0:	7547a783          	lw	a5,1876(a5) # 1c001754 <_rt_io_uart>
1c00b7b4:	c391                	beqz	a5,1c00b7b8 <tfp_putc.isra.8+0xc>
1c00b7b6:	bf7d                	j	1c00b774 <__rt_putc_uart>
1c00b7b8:	1c0017b7          	lui	a5,0x1c001
1c00b7bc:	5947a783          	lw	a5,1428(a5) # 1c001594 <__hal_debug_struct+0x10>
1c00b7c0:	c395                	beqz	a5,1c00b7e4 <tfp_putc.isra.8+0x38>
1c00b7c2:	6689                	lui	a3,0x2
1c00b7c4:	f14027f3          	csrr	a5,mhartid
1c00b7c8:	f8068693          	addi	a3,a3,-128 # 1f80 <__rt_hyper_pending_tasks_last+0x1a18>
1c00b7cc:	00379713          	slli	a4,a5,0x3
1c00b7d0:	078a                	slli	a5,a5,0x2
1c00b7d2:	ee873733          	p.bclr	a4,a4,23,8
1c00b7d6:	8ff5                	and	a5,a5,a3
1c00b7d8:	97ba                	add	a5,a5,a4
1c00b7da:	1a120737          	lui	a4,0x1a120
1c00b7de:	00a767a3          	p.sw	a0,a5(a4)
1c00b7e2:	8082                	ret
1c00b7e4:	bf0d                	j	1c00b716 <__rt_putc_debug_bridge>

1c00b7e6 <fputc_locked>:
1c00b7e6:	1141                	addi	sp,sp,-16
1c00b7e8:	c422                	sw	s0,8(sp)
1c00b7ea:	842a                	mv	s0,a0
1c00b7ec:	0ff57513          	andi	a0,a0,255
1c00b7f0:	c606                	sw	ra,12(sp)
1c00b7f2:	3f6d                	jal	1c00b7ac <tfp_putc.isra.8>
1c00b7f4:	8522                	mv	a0,s0
1c00b7f6:	40b2                	lw	ra,12(sp)
1c00b7f8:	4422                	lw	s0,8(sp)
1c00b7fa:	0141                	addi	sp,sp,16
1c00b7fc:	8082                	ret

1c00b7fe <_prf_locked>:
1c00b7fe:	1101                	addi	sp,sp,-32
1c00b800:	ce06                	sw	ra,28(sp)
1c00b802:	c02a                	sw	a0,0(sp)
1c00b804:	c62e                	sw	a1,12(sp)
1c00b806:	c432                	sw	a2,8(sp)
1c00b808:	c236                	sw	a3,4(sp)
1c00b80a:	c59ff0ef          	jal	ra,1c00b462 <__rt_io_lock>
1c00b80e:	4692                	lw	a3,4(sp)
1c00b810:	4622                	lw	a2,8(sp)
1c00b812:	45b2                	lw	a1,12(sp)
1c00b814:	4502                	lw	a0,0(sp)
1c00b816:	22c5                	jal	1c00b9f6 <_prf>
1c00b818:	c02a                	sw	a0,0(sp)
1c00b81a:	3971                	jal	1c00b4b6 <__rt_io_unlock>
1c00b81c:	40f2                	lw	ra,28(sp)
1c00b81e:	4502                	lw	a0,0(sp)
1c00b820:	6105                	addi	sp,sp,32
1c00b822:	8082                	ret

1c00b824 <exit>:
1c00b824:	1141                	addi	sp,sp,-16
1c00b826:	c422                	sw	s0,8(sp)
1c00b828:	1a104437          	lui	s0,0x1a104
1c00b82c:	0a040793          	addi	a5,s0,160 # 1a1040a0 <__l1_end+0xa100088>
1c00b830:	c606                	sw	ra,12(sp)
1c00b832:	c226                	sw	s1,4(sp)
1c00b834:	c04a                	sw	s2,0(sp)
1c00b836:	1c0014b7          	lui	s1,0x1c001
1c00b83a:	c1f54933          	p.bset	s2,a0,0,31
1c00b83e:	0127a023          	sw	s2,0(a5)
1c00b842:	58448493          	addi	s1,s1,1412 # 1c001584 <__hal_debug_struct>
1c00b846:	da4ff0ef          	jal	ra,1c00adea <__rt_bridge_printf_flush>
1c00b84a:	0124a623          	sw	s2,12(s1)
1c00b84e:	d5eff0ef          	jal	ra,1c00adac <__rt_bridge_send_notif>
1c00b852:	449c                	lw	a5,8(s1)
1c00b854:	cb91                	beqz	a5,1c00b868 <exit+0x44>
1c00b856:	07440413          	addi	s0,s0,116
1c00b85a:	401c                	lw	a5,0(s0)
1c00b85c:	cc9797b3          	p.extractu	a5,a5,6,9
1c00b860:	fe77bde3          	p.bneimm	a5,7,1c00b85a <exit+0x36>
1c00b864:	d68ff0ef          	jal	ra,1c00adcc <__rt_bridge_clear_notif>
1c00b868:	a001                	j	1c00b868 <exit+0x44>

1c00b86a <abort>:
1c00b86a:	1141                	addi	sp,sp,-16
1c00b86c:	557d                	li	a0,-1
1c00b86e:	c606                	sw	ra,12(sp)
1c00b870:	3f55                	jal	1c00b824 <exit>

1c00b872 <__rt_io_init>:
1c00b872:	1c0017b7          	lui	a5,0x1c001
1c00b876:	57478793          	addi	a5,a5,1396 # 1c001574 <__rt_io_fc_lock>
1c00b87a:	0007a223          	sw	zero,4(a5)
1c00b87e:	0007a023          	sw	zero,0(a5)
1c00b882:	0007a623          	sw	zero,12(a5)
1c00b886:	1c0017b7          	lui	a5,0x1c001
1c00b88a:	7407aa23          	sw	zero,1876(a5) # 1c001754 <_rt_io_uart>
1c00b88e:	1c0017b7          	lui	a5,0x1c001
1c00b892:	7407a623          	sw	zero,1868(a5) # 1c00174c <__rt_io_event_current>
1c00b896:	1c0017b7          	lui	a5,0x1c001
1c00b89a:	7587a783          	lw	a5,1880(a5) # 1c001758 <__rt_iodev>
1c00b89e:	0217be63          	p.bneimm	a5,1,1c00b8da <__rt_io_init+0x68>
1c00b8a2:	1c00b5b7          	lui	a1,0x1c00b
1c00b8a6:	1141                	addi	sp,sp,-16
1c00b8a8:	4601                	li	a2,0
1c00b8aa:	3e058593          	addi	a1,a1,992 # 1c00b3e0 <__rt_io_start>
1c00b8ae:	4501                	li	a0,0
1c00b8b0:	c606                	sw	ra,12(sp)
1c00b8b2:	a2eff0ef          	jal	ra,1c00aae0 <__rt_cbsys_add>
1c00b8b6:	1c00b5b7          	lui	a1,0x1c00b
1c00b8ba:	59858593          	addi	a1,a1,1432 # 1c00b598 <__rt_io_stop>
1c00b8be:	4601                	li	a2,0
1c00b8c0:	4505                	li	a0,1
1c00b8c2:	a1eff0ef          	jal	ra,1c00aae0 <__rt_cbsys_add>
1c00b8c6:	40b2                	lw	ra,12(sp)
1c00b8c8:	1c0017b7          	lui	a5,0x1c001
1c00b8cc:	7407a823          	sw	zero,1872(a5) # 1c001750 <__rt_io_pending_flush>
1c00b8d0:	4585                	li	a1,1
1c00b8d2:	4501                	li	a0,0
1c00b8d4:	0141                	addi	sp,sp,16
1c00b8d6:	d13fd06f          	j	1c0095e8 <rt_event_alloc>
1c00b8da:	8082                	ret

1c00b8dc <printf>:
1c00b8dc:	7139                	addi	sp,sp,-64
1c00b8de:	d432                	sw	a2,40(sp)
1c00b8e0:	862a                	mv	a2,a0
1c00b8e2:	1c00b537          	lui	a0,0x1c00b
1c00b8e6:	d22e                	sw	a1,36(sp)
1c00b8e8:	d636                	sw	a3,44(sp)
1c00b8ea:	4589                	li	a1,2
1c00b8ec:	1054                	addi	a3,sp,36
1c00b8ee:	7e650513          	addi	a0,a0,2022 # 1c00b7e6 <fputc_locked>
1c00b8f2:	ce06                	sw	ra,28(sp)
1c00b8f4:	d83a                	sw	a4,48(sp)
1c00b8f6:	da3e                	sw	a5,52(sp)
1c00b8f8:	dc42                	sw	a6,56(sp)
1c00b8fa:	de46                	sw	a7,60(sp)
1c00b8fc:	c636                	sw	a3,12(sp)
1c00b8fe:	3701                	jal	1c00b7fe <_prf_locked>
1c00b900:	40f2                	lw	ra,28(sp)
1c00b902:	6121                	addi	sp,sp,64
1c00b904:	8082                	ret

1c00b906 <_to_x>:
1c00b906:	872a                	mv	a4,a0
1c00b908:	87aa                	mv	a5,a0
1c00b90a:	4325                	li	t1,9
1c00b90c:	02c5f8b3          	remu	a7,a1,a2
1c00b910:	02700513          	li	a0,39
1c00b914:	02c5d5b3          	divu	a1,a1,a2
1c00b918:	0ff8f813          	andi	a6,a7,255
1c00b91c:	01136363          	bltu	t1,a7,1c00b922 <_to_x+0x1c>
1c00b920:	4501                	li	a0,0
1c00b922:	03080813          	addi	a6,a6,48
1c00b926:	9542                	add	a0,a0,a6
1c00b928:	00a780ab          	p.sb	a0,1(a5!)
1c00b92c:	f1e5                	bnez	a1,1c00b90c <_to_x+0x6>
1c00b92e:	03000613          	li	a2,48
1c00b932:	40e78533          	sub	a0,a5,a4
1c00b936:	00d54763          	blt	a0,a3,1c00b944 <_to_x+0x3e>
1c00b93a:	fe078fab          	p.sb	zero,-1(a5!)
1c00b93e:	00f76663          	bltu	a4,a5,1c00b94a <_to_x+0x44>
1c00b942:	8082                	ret
1c00b944:	00c780ab          	p.sb	a2,1(a5!)
1c00b948:	b7ed                	j	1c00b932 <_to_x+0x2c>
1c00b94a:	00074603          	lbu	a2,0(a4) # 1a120000 <__l1_end+0xa11bfe8>
1c00b94e:	0007c683          	lbu	a3,0(a5)
1c00b952:	fec78fab          	p.sb	a2,-1(a5!)
1c00b956:	00d700ab          	p.sb	a3,1(a4!)
1c00b95a:	b7d5                	j	1c00b93e <_to_x+0x38>

1c00b95c <_rlrshift>:
1c00b95c:	411c                	lw	a5,0(a0)
1c00b95e:	4154                	lw	a3,4(a0)
1c00b960:	fc17b733          	p.bclr	a4,a5,30,1
1c00b964:	01f69613          	slli	a2,a3,0x1f
1c00b968:	8385                	srli	a5,a5,0x1
1c00b96a:	8fd1                	or	a5,a5,a2
1c00b96c:	97ba                	add	a5,a5,a4
1c00b96e:	8285                	srli	a3,a3,0x1
1c00b970:	00e7b733          	sltu	a4,a5,a4
1c00b974:	9736                	add	a4,a4,a3
1c00b976:	c11c                	sw	a5,0(a0)
1c00b978:	c158                	sw	a4,4(a0)
1c00b97a:	8082                	ret

1c00b97c <_ldiv5>:
1c00b97c:	4118                	lw	a4,0(a0)
1c00b97e:	4154                	lw	a3,4(a0)
1c00b980:	4615                	li	a2,5
1c00b982:	00270793          	addi	a5,a4,2
1c00b986:	00e7b733          	sltu	a4,a5,a4
1c00b98a:	9736                	add	a4,a4,a3
1c00b98c:	02c755b3          	divu	a1,a4,a2
1c00b990:	42b61733          	p.msu	a4,a2,a1
1c00b994:	01d71693          	slli	a3,a4,0x1d
1c00b998:	0037d713          	srli	a4,a5,0x3
1c00b99c:	8f55                	or	a4,a4,a3
1c00b99e:	02c75733          	divu	a4,a4,a2
1c00b9a2:	01d75693          	srli	a3,a4,0x1d
1c00b9a6:	070e                	slli	a4,a4,0x3
1c00b9a8:	42e617b3          	p.msu	a5,a2,a4
1c00b9ac:	95b6                	add	a1,a1,a3
1c00b9ae:	02c7d7b3          	divu	a5,a5,a2
1c00b9b2:	973e                	add	a4,a4,a5
1c00b9b4:	00f737b3          	sltu	a5,a4,a5
1c00b9b8:	97ae                	add	a5,a5,a1
1c00b9ba:	c118                	sw	a4,0(a0)
1c00b9bc:	c15c                	sw	a5,4(a0)
1c00b9be:	8082                	ret

1c00b9c0 <_get_digit>:
1c00b9c0:	419c                	lw	a5,0(a1)
1c00b9c2:	03000713          	li	a4,48
1c00b9c6:	02f05563          	blez	a5,1c00b9f0 <_get_digit+0x30>
1c00b9ca:	17fd                	addi	a5,a5,-1
1c00b9cc:	c19c                	sw	a5,0(a1)
1c00b9ce:	411c                	lw	a5,0(a0)
1c00b9d0:	4729                	li	a4,10
1c00b9d2:	4150                	lw	a2,4(a0)
1c00b9d4:	02f706b3          	mul	a3,a4,a5
1c00b9d8:	02f737b3          	mulhu	a5,a4,a5
1c00b9dc:	c114                	sw	a3,0(a0)
1c00b9de:	42c707b3          	p.mac	a5,a4,a2
1c00b9e2:	01c7d713          	srli	a4,a5,0x1c
1c00b9e6:	c7c7b7b3          	p.bclr	a5,a5,3,28
1c00b9ea:	03070713          	addi	a4,a4,48
1c00b9ee:	c15c                	sw	a5,4(a0)
1c00b9f0:	0ff77513          	andi	a0,a4,255
1c00b9f4:	8082                	ret

1c00b9f6 <_prf>:
1c00b9f6:	714d                	addi	sp,sp,-336
1c00b9f8:	14912223          	sw	s1,324(sp)
1c00b9fc:	15212023          	sw	s2,320(sp)
1c00ba00:	13812423          	sw	s8,296(sp)
1c00ba04:	14112623          	sw	ra,332(sp)
1c00ba08:	14812423          	sw	s0,328(sp)
1c00ba0c:	13312e23          	sw	s3,316(sp)
1c00ba10:	13412c23          	sw	s4,312(sp)
1c00ba14:	13512a23          	sw	s5,308(sp)
1c00ba18:	13612823          	sw	s6,304(sp)
1c00ba1c:	13712623          	sw	s7,300(sp)
1c00ba20:	13912223          	sw	s9,292(sp)
1c00ba24:	13a12023          	sw	s10,288(sp)
1c00ba28:	11b12e23          	sw	s11,284(sp)
1c00ba2c:	cc2a                	sw	a0,24(sp)
1c00ba2e:	ce2e                	sw	a1,28(sp)
1c00ba30:	84b2                	mv	s1,a2
1c00ba32:	8c36                	mv	s8,a3
1c00ba34:	4901                	li	s2,0
1c00ba36:	0004c503          	lbu	a0,0(s1)
1c00ba3a:	00148b93          	addi	s7,s1,1
1c00ba3e:	c919                	beqz	a0,1c00ba54 <_prf+0x5e>
1c00ba40:	02500793          	li	a5,37
1c00ba44:	14f50763          	beq	a0,a5,1c00bb92 <_prf+0x19c>
1c00ba48:	45f2                	lw	a1,28(sp)
1c00ba4a:	4762                	lw	a4,24(sp)
1c00ba4c:	9702                	jalr	a4
1c00ba4e:	05f53063          	p.bneimm	a0,-1,1c00ba8e <_prf+0x98>
1c00ba52:	597d                	li	s2,-1
1c00ba54:	14c12083          	lw	ra,332(sp)
1c00ba58:	14812403          	lw	s0,328(sp)
1c00ba5c:	854a                	mv	a0,s2
1c00ba5e:	14412483          	lw	s1,324(sp)
1c00ba62:	14012903          	lw	s2,320(sp)
1c00ba66:	13c12983          	lw	s3,316(sp)
1c00ba6a:	13812a03          	lw	s4,312(sp)
1c00ba6e:	13412a83          	lw	s5,308(sp)
1c00ba72:	13012b03          	lw	s6,304(sp)
1c00ba76:	12c12b83          	lw	s7,300(sp)
1c00ba7a:	12812c03          	lw	s8,296(sp)
1c00ba7e:	12412c83          	lw	s9,292(sp)
1c00ba82:	12012d03          	lw	s10,288(sp)
1c00ba86:	11c12d83          	lw	s11,284(sp)
1c00ba8a:	6171                	addi	sp,sp,336
1c00ba8c:	8082                	ret
1c00ba8e:	0905                	addi	s2,s2,1
1c00ba90:	8a62                	mv	s4,s8
1c00ba92:	84de                	mv	s1,s7
1c00ba94:	8c52                	mv	s8,s4
1c00ba96:	b745                	j	1c00ba36 <_prf+0x40>
1c00ba98:	0f3a8463          	beq	s5,s3,1c00bb80 <_prf+0x18a>
1c00ba9c:	0d59e763          	bltu	s3,s5,1c00bb6a <_prf+0x174>
1c00baa0:	fa0a8ae3          	beqz	s5,1c00ba54 <_prf+0x5e>
1c00baa4:	0dba8c63          	beq	s5,s11,1c00bb7c <_prf+0x186>
1c00baa8:	8ba6                	mv	s7,s1
1c00baaa:	000bca83          	lbu	s5,0(s7)
1c00baae:	1c0017b7          	lui	a5,0x1c001
1c00bab2:	b9078513          	addi	a0,a5,-1136 # 1c000b90 <PIo2+0x234>
1c00bab6:	85d6                	mv	a1,s5
1c00bab8:	001b8493          	addi	s1,s7,1
1c00babc:	c41ff0ef          	jal	ra,1c00b6fc <strchr>
1c00bac0:	fd61                	bnez	a0,1c00ba98 <_prf+0xa2>
1c00bac2:	02a00693          	li	a3,42
1c00bac6:	0eda9863          	bne	s5,a3,1c00bbb6 <_prf+0x1c0>
1c00baca:	000c2983          	lw	s3,0(s8)
1c00bace:	004c0693          	addi	a3,s8,4
1c00bad2:	0009d663          	bgez	s3,1c00bade <_prf+0xe8>
1c00bad6:	4705                	li	a4,1
1c00bad8:	413009b3          	neg	s3,s3
1c00badc:	ca3a                	sw	a4,20(sp)
1c00bade:	0004ca83          	lbu	s5,0(s1)
1c00bae2:	8c36                	mv	s8,a3
1c00bae4:	002b8493          	addi	s1,s7,2
1c00bae8:	0c800713          	li	a4,200
1c00baec:	02e00693          	li	a3,46
1c00baf0:	04e9d9b3          	p.minu	s3,s3,a4
1c00baf4:	5d7d                	li	s10,-1
1c00baf6:	02da9463          	bne	s5,a3,1c00bb1e <_prf+0x128>
1c00bafa:	0004c703          	lbu	a4,0(s1)
1c00bafe:	02a00793          	li	a5,42
1c00bb02:	0ef71d63          	bne	a4,a5,1c00bbfc <_prf+0x206>
1c00bb06:	000c2d03          	lw	s10,0(s8)
1c00bb0a:	0485                	addi	s1,s1,1
1c00bb0c:	0c11                	addi	s8,s8,4
1c00bb0e:	0c800793          	li	a5,200
1c00bb12:	01a7d363          	ble	s10,a5,1c00bb18 <_prf+0x122>
1c00bb16:	5d7d                	li	s10,-1
1c00bb18:	0004ca83          	lbu	s5,0(s1)
1c00bb1c:	0485                	addi	s1,s1,1
1c00bb1e:	1c0017b7          	lui	a5,0x1c001
1c00bb22:	85d6                	mv	a1,s5
1c00bb24:	b9878513          	addi	a0,a5,-1128 # 1c000b98 <PIo2+0x23c>
1c00bb28:	bd5ff0ef          	jal	ra,1c00b6fc <strchr>
1c00bb2c:	c501                	beqz	a0,1c00bb34 <_prf+0x13e>
1c00bb2e:	0004ca83          	lbu	s5,0(s1)
1c00bb32:	0485                	addi	s1,s1,1
1c00bb34:	06700693          	li	a3,103
1c00bb38:	1356c563          	blt	a3,s5,1c00bc62 <_prf+0x26c>
1c00bb3c:	06500693          	li	a3,101
1c00bb40:	20dad163          	ble	a3,s5,1c00bd42 <_prf+0x34c>
1c00bb44:	04700693          	li	a3,71
1c00bb48:	0b56ce63          	blt	a3,s5,1c00bc04 <_prf+0x20e>
1c00bb4c:	04500693          	li	a3,69
1c00bb50:	1edad963          	ble	a3,s5,1c00bd42 <_prf+0x34c>
1c00bb54:	f00a80e3          	beqz	s5,1c00ba54 <_prf+0x5e>
1c00bb58:	02500713          	li	a4,37
1c00bb5c:	64ea8d63          	beq	s5,a4,1c00c1b6 <_prf+0x7c0>
1c00bb60:	0c800713          	li	a4,200
1c00bb64:	67575163          	ble	s5,a4,1c00c1c6 <_prf+0x7d0>
1c00bb68:	b5ed                	j	1c00ba52 <_prf+0x5c>
1c00bb6a:	034a8163          	beq	s5,s4,1c00bb8c <_prf+0x196>
1c00bb6e:	016a8b63          	beq	s5,s6,1c00bb84 <_prf+0x18e>
1c00bb72:	f3aa9be3          	bne	s5,s10,1c00baa8 <_prf+0xb2>
1c00bb76:	4785                	li	a5,1
1c00bb78:	c83e                	sw	a5,16(sp)
1c00bb7a:	b73d                	j	1c00baa8 <_prf+0xb2>
1c00bb7c:	4405                	li	s0,1
1c00bb7e:	b72d                	j	1c00baa8 <_prf+0xb2>
1c00bb80:	4c85                	li	s9,1
1c00bb82:	b71d                	j	1c00baa8 <_prf+0xb2>
1c00bb84:	03000713          	li	a4,48
1c00bb88:	c63a                	sw	a4,12(sp)
1c00bb8a:	bf39                	j	1c00baa8 <_prf+0xb2>
1c00bb8c:	4785                	li	a5,1
1c00bb8e:	ca3e                	sw	a5,20(sp)
1c00bb90:	bf21                	j	1c00baa8 <_prf+0xb2>
1c00bb92:	02000713          	li	a4,32
1c00bb96:	c63a                	sw	a4,12(sp)
1c00bb98:	4401                	li	s0,0
1c00bb9a:	c802                	sw	zero,16(sp)
1c00bb9c:	ca02                	sw	zero,20(sp)
1c00bb9e:	4c81                	li	s9,0
1c00bba0:	02300993          	li	s3,35
1c00bba4:	02d00a13          	li	s4,45
1c00bba8:	03000b13          	li	s6,48
1c00bbac:	02b00d13          	li	s10,43
1c00bbb0:	02000d93          	li	s11,32
1c00bbb4:	bddd                	j	1c00baaa <_prf+0xb4>
1c00bbb6:	fd0a8693          	addi	a3,s5,-48
1c00bbba:	4625                	li	a2,9
1c00bbbc:	4981                	li	s3,0
1c00bbbe:	f2d665e3          	bltu	a2,a3,1c00bae8 <_prf+0xf2>
1c00bbc2:	46a5                	li	a3,9
1c00bbc4:	45a9                	li	a1,10
1c00bbc6:	84de                	mv	s1,s7
1c00bbc8:	0014c70b          	p.lbu	a4,1(s1!)
1c00bbcc:	fd070613          	addi	a2,a4,-48
1c00bbd0:	8aba                	mv	s5,a4
1c00bbd2:	f0c6ebe3          	bltu	a3,a2,1c00bae8 <_prf+0xf2>
1c00bbd6:	42b98733          	p.mac	a4,s3,a1
1c00bbda:	8ba6                	mv	s7,s1
1c00bbdc:	fd070993          	addi	s3,a4,-48
1c00bbe0:	b7dd                	j	1c00bbc6 <_prf+0x1d0>
1c00bbe2:	42ad07b3          	p.mac	a5,s10,a0
1c00bbe6:	84b6                	mv	s1,a3
1c00bbe8:	fd078d13          	addi	s10,a5,-48
1c00bbec:	86a6                	mv	a3,s1
1c00bbee:	0016c78b          	p.lbu	a5,1(a3!)
1c00bbf2:	fd078593          	addi	a1,a5,-48
1c00bbf6:	feb676e3          	bleu	a1,a2,1c00bbe2 <_prf+0x1ec>
1c00bbfa:	bf11                	j	1c00bb0e <_prf+0x118>
1c00bbfc:	4d01                	li	s10,0
1c00bbfe:	4625                	li	a2,9
1c00bc00:	4529                	li	a0,10
1c00bc02:	b7ed                	j	1c00bbec <_prf+0x1f6>
1c00bc04:	06300693          	li	a3,99
1c00bc08:	0cda8e63          	beq	s5,a3,1c00bce4 <_prf+0x2ee>
1c00bc0c:	0756cb63          	blt	a3,s5,1c00bc82 <_prf+0x28c>
1c00bc10:	05800693          	li	a3,88
1c00bc14:	f4da96e3          	bne	s5,a3,1c00bb60 <_prf+0x16a>
1c00bc18:	04410b93          	addi	s7,sp,68
1c00bc1c:	004c0a13          	addi	s4,s8,4
1c00bc20:	000c2583          	lw	a1,0(s8)
1c00bc24:	845e                	mv	s0,s7
1c00bc26:	000c8963          	beqz	s9,1c00bc38 <_prf+0x242>
1c00bc2a:	76e1                	lui	a3,0xffff8
1c00bc2c:	8306c693          	xori	a3,a3,-2000
1c00bc30:	04d11223          	sh	a3,68(sp)
1c00bc34:	04610413          	addi	s0,sp,70
1c00bc38:	86ea                	mv	a3,s10
1c00bc3a:	4641                	li	a2,16
1c00bc3c:	8522                	mv	a0,s0
1c00bc3e:	31e1                	jal	1c00b906 <_to_x>
1c00bc40:	05800693          	li	a3,88
1c00bc44:	00da9863          	bne	s5,a3,1c00bc54 <_prf+0x25e>
1c00bc48:	86de                	mv	a3,s7
1c00bc4a:	45e5                	li	a1,25
1c00bc4c:	0016c78b          	p.lbu	a5,1(a3!) # ffff8001 <pulp__FC+0xffff8002>
1c00bc50:	54079963          	bnez	a5,1c00c1a2 <_prf+0x7ac>
1c00bc54:	41740433          	sub	s0,s0,s7
1c00bc58:	9522                	add	a0,a0,s0
1c00bc5a:	01903433          	snez	s0,s9
1c00bc5e:	0406                	slli	s0,s0,0x1
1c00bc60:	a0f1                	j	1c00bd2c <_prf+0x336>
1c00bc62:	07000693          	li	a3,112
1c00bc66:	4eda8763          	beq	s5,a3,1c00c154 <_prf+0x75e>
1c00bc6a:	0556c163          	blt	a3,s5,1c00bcac <_prf+0x2b6>
1c00bc6e:	06e00693          	li	a3,110
1c00bc72:	46da8463          	beq	s5,a3,1c00c0da <_prf+0x6e4>
1c00bc76:	4756c963          	blt	a3,s5,1c00c0e8 <_prf+0x6f2>
1c00bc7a:	06900693          	li	a3,105
1c00bc7e:	eeda91e3          	bne	s5,a3,1c00bb60 <_prf+0x16a>
1c00bc82:	000c2a83          	lw	s5,0(s8)
1c00bc86:	004c0a13          	addi	s4,s8,4
1c00bc8a:	04410b13          	addi	s6,sp,68
1c00bc8e:	060ad663          	bgez	s5,1c00bcfa <_prf+0x304>
1c00bc92:	02d00693          	li	a3,45
1c00bc96:	04d10223          	sb	a3,68(sp)
1c00bc9a:	80000737          	lui	a4,0x80000
1c00bc9e:	415005b3          	neg	a1,s5
1c00bca2:	06ea9663          	bne	s5,a4,1c00bd0e <_prf+0x318>
1c00bca6:	800005b7          	lui	a1,0x80000
1c00bcaa:	a095                	j	1c00bd0e <_prf+0x318>
1c00bcac:	07500693          	li	a3,117
1c00bcb0:	4cda8f63          	beq	s5,a3,1c00c18e <_prf+0x798>
1c00bcb4:	07800693          	li	a3,120
1c00bcb8:	f6da80e3          	beq	s5,a3,1c00bc18 <_prf+0x222>
1c00bcbc:	07300693          	li	a3,115
1c00bcc0:	eada90e3          	bne	s5,a3,1c00bb60 <_prf+0x16a>
1c00bcc4:	000c2583          	lw	a1,0(s8)
1c00bcc8:	004c0a13          	addi	s4,s8,4
1c00bccc:	4c81                	li	s9,0
1c00bcce:	86ae                	mv	a3,a1
1c00bcd0:	0c8350fb          	lp.setupi	x1,200,1c00bcdc <_prf+0x2e6>
1c00bcd4:	0016c60b          	p.lbu	a2,1(a3!)
1c00bcd8:	4a060063          	beqz	a2,1c00c178 <_prf+0x782>
1c00bcdc:	0c85                	addi	s9,s9,1
1c00bcde:	480d5f63          	bgez	s10,1c00c17c <_prf+0x786>
1c00bce2:	a14d                	j	1c00c184 <_prf+0x78e>
1c00bce4:	000c2783          	lw	a5,0(s8)
1c00bce8:	004c0a13          	addi	s4,s8,4
1c00bcec:	040102a3          	sb	zero,69(sp)
1c00bcf0:	04f10223          	sb	a5,68(sp)
1c00bcf4:	4c85                	li	s9,1
1c00bcf6:	4401                	li	s0,0
1c00bcf8:	a919                	j	1c00c10e <_prf+0x718>
1c00bcfa:	47c2                	lw	a5,16(sp)
1c00bcfc:	02b00693          	li	a3,43
1c00bd00:	e781                	bnez	a5,1c00bd08 <_prf+0x312>
1c00bd02:	c81d                	beqz	s0,1c00bd38 <_prf+0x342>
1c00bd04:	02000693          	li	a3,32
1c00bd08:	04d10223          	sb	a3,68(sp)
1c00bd0c:	85d6                	mv	a1,s5
1c00bd0e:	04510c13          	addi	s8,sp,69
1c00bd12:	86ea                	mv	a3,s10
1c00bd14:	4629                	li	a2,10
1c00bd16:	8562                	mv	a0,s8
1c00bd18:	befff0ef          	jal	ra,1c00b906 <_to_x>
1c00bd1c:	4742                	lw	a4,16(sp)
1c00bd1e:	9562                	add	a0,a0,s8
1c00bd20:	41650533          	sub	a0,a0,s6
1c00bd24:	ef09                	bnez	a4,1c00bd3e <_prf+0x348>
1c00bd26:	e019                	bnez	s0,1c00bd2c <_prf+0x336>
1c00bd28:	01fad413          	srli	s0,s5,0x1f
1c00bd2c:	0bfd2363          	p.beqimm	s10,-1,1c00bdd2 <_prf+0x3dc>
1c00bd30:	02000713          	li	a4,32
1c00bd34:	c63a                	sw	a4,12(sp)
1c00bd36:	a871                	j	1c00bdd2 <_prf+0x3dc>
1c00bd38:	85d6                	mv	a1,s5
1c00bd3a:	8c5a                	mv	s8,s6
1c00bd3c:	bfd9                	j	1c00bd12 <_prf+0x31c>
1c00bd3e:	4442                	lw	s0,16(sp)
1c00bd40:	b7f5                	j	1c00bd2c <_prf+0x336>
1c00bd42:	0c1d                	addi	s8,s8,7
1c00bd44:	c40c3c33          	p.bclr	s8,s8,2,0
1c00bd48:	000c2883          	lw	a7,0(s8)
1c00bd4c:	004c2303          	lw	t1,4(s8)
1c00bd50:	800007b7          	lui	a5,0x80000
1c00bd54:	0158d593          	srli	a1,a7,0x15
1c00bd58:	00b31693          	slli	a3,t1,0xb
1c00bd5c:	8ecd                	or	a3,a3,a1
1c00bd5e:	fff7c793          	not	a5,a5
1c00bd62:	01435613          	srli	a2,t1,0x14
1c00bd66:	08ae                	slli	a7,a7,0xb
1c00bd68:	8efd                	and	a3,a3,a5
1c00bd6a:	e8b63633          	p.bclr	a2,a2,20,11
1c00bd6e:	d846                	sw	a7,48(sp)
1c00bd70:	da36                	sw	a3,52(sp)
1c00bd72:	7ff00593          	li	a1,2047
1c00bd76:	008c0a13          	addi	s4,s8,8
1c00bd7a:	08b61d63          	bne	a2,a1,1c00be14 <_prf+0x41e>
1c00bd7e:	00d0                	addi	a2,sp,68
1c00bd80:	8732                	mv	a4,a2
1c00bd82:	00035863          	bgez	t1,1c00bd92 <_prf+0x39c>
1c00bd86:	02d00713          	li	a4,45
1c00bd8a:	04e10223          	sb	a4,68(sp)
1c00bd8e:	04510713          	addi	a4,sp,69
1c00bd92:	00d8e6b3          	or	a3,a7,a3
1c00bd96:	fbfa8793          	addi	a5,s5,-65
1c00bd9a:	00370513          	addi	a0,a4,3 # 80000003 <pulp__FC+0x80000004>
1c00bd9e:	eaa1                	bnez	a3,1c00bdee <_prf+0x3f8>
1c00bda0:	46e5                	li	a3,25
1c00bda2:	02f6ee63          	bltu	a3,a5,1c00bdde <_prf+0x3e8>
1c00bda6:	6795                	lui	a5,0x5
1c00bda8:	e4978793          	addi	a5,a5,-439 # 4e49 <_l1_preload_size+0xe39>
1c00bdac:	00f71023          	sh	a5,0(a4)
1c00bdb0:	04600793          	li	a5,70
1c00bdb4:	00f70123          	sb	a5,2(a4)
1c00bdb8:	000701a3          	sb	zero,3(a4)
1c00bdbc:	8d11                	sub	a0,a0,a2
1c00bdbe:	47c2                	lw	a5,16(sp)
1c00bdc0:	46079163          	bnez	a5,1c00c222 <_prf+0x82c>
1c00bdc4:	e419                	bnez	s0,1c00bdd2 <_prf+0x3dc>
1c00bdc6:	04414403          	lbu	s0,68(sp)
1c00bdca:	fd340413          	addi	s0,s0,-45
1c00bdce:	00143413          	seqz	s0,s0
1c00bdd2:	0c800793          	li	a5,200
1c00bdd6:	c6a7cee3          	blt	a5,a0,1c00ba52 <_prf+0x5c>
1c00bdda:	8caa                	mv	s9,a0
1c00bddc:	ae0d                	j	1c00c10e <_prf+0x718>
1c00bdde:	679d                	lui	a5,0x7
1c00bde0:	e6978793          	addi	a5,a5,-407 # 6e69 <_l1_preload_size+0x2e59>
1c00bde4:	00f71023          	sh	a5,0(a4)
1c00bde8:	06600793          	li	a5,102
1c00bdec:	b7e1                	j	1c00bdb4 <_prf+0x3be>
1c00bdee:	46e5                	li	a3,25
1c00bdf0:	00f6ea63          	bltu	a3,a5,1c00be04 <_prf+0x40e>
1c00bdf4:	6791                	lui	a5,0x4
1c00bdf6:	14e78793          	addi	a5,a5,334 # 414e <_l1_preload_size+0x13e>
1c00bdfa:	00f71023          	sh	a5,0(a4)
1c00bdfe:	04e00793          	li	a5,78
1c00be02:	bf4d                	j	1c00bdb4 <_prf+0x3be>
1c00be04:	6799                	lui	a5,0x6
1c00be06:	16e78793          	addi	a5,a5,366 # 616e <_l1_preload_size+0x215e>
1c00be0a:	00f71023          	sh	a5,0(a4)
1c00be0e:	06e00793          	li	a5,110
1c00be12:	b74d                	j	1c00bdb4 <_prf+0x3be>
1c00be14:	04600593          	li	a1,70
1c00be18:	00ba9463          	bne	s5,a1,1c00be20 <_prf+0x42a>
1c00be1c:	06600a93          	li	s5,102
1c00be20:	011665b3          	or	a1,a2,a7
1c00be24:	8dd5                	or	a1,a1,a3
1c00be26:	c5d9                	beqz	a1,1c00beb4 <_prf+0x4be>
1c00be28:	80000737          	lui	a4,0x80000
1c00be2c:	8ed9                	or	a3,a3,a4
1c00be2e:	da36                	sw	a3,52(sp)
1c00be30:	d846                	sw	a7,48(sp)
1c00be32:	c0260c13          	addi	s8,a2,-1022
1c00be36:	02d00693          	li	a3,45
1c00be3a:	00034b63          	bltz	t1,1c00be50 <_prf+0x45a>
1c00be3e:	47c2                	lw	a5,16(sp)
1c00be40:	02b00693          	li	a3,43
1c00be44:	e791                	bnez	a5,1c00be50 <_prf+0x45a>
1c00be46:	04410b13          	addi	s6,sp,68
1c00be4a:	c419                	beqz	s0,1c00be58 <_prf+0x462>
1c00be4c:	02000693          	li	a3,32
1c00be50:	04d10223          	sb	a3,68(sp)
1c00be54:	04510b13          	addi	s6,sp,69
1c00be58:	4b81                	li	s7,0
1c00be5a:	55f9                	li	a1,-2
1c00be5c:	06bc4163          	blt	s8,a1,1c00bebe <_prf+0x4c8>
1c00be60:	0b804763          	bgtz	s8,1c00bf0e <_prf+0x518>
1c00be64:	1808                	addi	a0,sp,48
1c00be66:	0c05                	addi	s8,s8,1
1c00be68:	af5ff0ef          	jal	ra,1c00b95c <_rlrshift>
1c00be6c:	fe4c3ce3          	p.bneimm	s8,4,1c00be64 <_prf+0x46e>
1c00be70:	000d5363          	bgez	s10,1c00be76 <_prf+0x480>
1c00be74:	4d19                	li	s10,6
1c00be76:	c05ab5b3          	p.bclr	a1,s5,0,5
1c00be7a:	04700513          	li	a0,71
1c00be7e:	0ca59463          	bne	a1,a0,1c00bf46 <_prf+0x550>
1c00be82:	4c01                	li	s8,0
1c00be84:	000c9463          	bnez	s9,1c00be8c <_prf+0x496>
1c00be88:	01a03c33          	snez	s8,s10
1c00be8c:	55f5                	li	a1,-3
1c00be8e:	00bbc663          	blt	s7,a1,1c00be9a <_prf+0x4a4>
1c00be92:	001d0593          	addi	a1,s10,1
1c00be96:	0b75dd63          	ble	s7,a1,1c00bf50 <_prf+0x55a>
1c00be9a:	06700593          	li	a1,103
1c00be9e:	14ba8863          	beq	s5,a1,1c00bfee <_prf+0x5f8>
1c00bea2:	04500a93          	li	s5,69
1c00bea6:	001d0593          	addi	a1,s10,1
1c00beaa:	4541                	li	a0,16
1c00beac:	d62a                	sw	a0,44(sp)
1c00beae:	04a5cdb3          	p.min	s11,a1,a0
1c00beb2:	a845                	j	1c00bf62 <_prf+0x56c>
1c00beb4:	4c01                	li	s8,0
1c00beb6:	b761                	j	1c00be3e <_prf+0x448>
1c00beb8:	1808                	addi	a0,sp,48
1c00beba:	aa3ff0ef          	jal	ra,1c00b95c <_rlrshift>
1c00bebe:	5352                	lw	t1,52(sp)
1c00bec0:	33333737          	lui	a4,0x33333
1c00bec4:	33270713          	addi	a4,a4,818 # 33333332 <__l2_shared_end+0x1731f192>
1c00bec8:	58c2                	lw	a7,48(sp)
1c00beca:	0c05                	addi	s8,s8,1
1c00becc:	fe6766e3          	bltu	a4,t1,1c00beb8 <_prf+0x4c2>
1c00bed0:	4515                	li	a0,5
1c00bed2:	031535b3          	mulhu	a1,a0,a7
1c00bed6:	1bfd                	addi	s7,s7,-1
1c00bed8:	031508b3          	mul	a7,a0,a7
1c00bedc:	426505b3          	p.mac	a1,a0,t1
1c00bee0:	d846                	sw	a7,48(sp)
1c00bee2:	4501                	li	a0,0
1c00bee4:	da2e                	sw	a1,52(sp)
1c00bee6:	800007b7          	lui	a5,0x80000
1c00beea:	fff7c793          	not	a5,a5
1c00beee:	00b7f663          	bleu	a1,a5,1c00befa <_prf+0x504>
1c00bef2:	d525                	beqz	a0,1c00be5a <_prf+0x464>
1c00bef4:	d846                	sw	a7,48(sp)
1c00bef6:	da2e                	sw	a1,52(sp)
1c00bef8:	b78d                	j	1c00be5a <_prf+0x464>
1c00befa:	01f8d313          	srli	t1,a7,0x1f
1c00befe:	00159513          	slli	a0,a1,0x1
1c00bf02:	00a365b3          	or	a1,t1,a0
1c00bf06:	0886                	slli	a7,a7,0x1
1c00bf08:	1c7d                	addi	s8,s8,-1
1c00bf0a:	4505                	li	a0,1
1c00bf0c:	bfe9                	j	1c00bee6 <_prf+0x4f0>
1c00bf0e:	1808                	addi	a0,sp,48
1c00bf10:	a6dff0ef          	jal	ra,1c00b97c <_ldiv5>
1c00bf14:	58c2                	lw	a7,48(sp)
1c00bf16:	55d2                	lw	a1,52(sp)
1c00bf18:	1c7d                	addi	s8,s8,-1
1c00bf1a:	0b85                	addi	s7,s7,1
1c00bf1c:	4501                	li	a0,0
1c00bf1e:	80000737          	lui	a4,0x80000
1c00bf22:	fff74713          	not	a4,a4
1c00bf26:	00b77663          	bleu	a1,a4,1c00bf32 <_prf+0x53c>
1c00bf2a:	d91d                	beqz	a0,1c00be60 <_prf+0x46a>
1c00bf2c:	d846                	sw	a7,48(sp)
1c00bf2e:	da2e                	sw	a1,52(sp)
1c00bf30:	bf05                	j	1c00be60 <_prf+0x46a>
1c00bf32:	01f8d313          	srli	t1,a7,0x1f
1c00bf36:	00159513          	slli	a0,a1,0x1
1c00bf3a:	00a365b3          	or	a1,t1,a0
1c00bf3e:	0886                	slli	a7,a7,0x1
1c00bf40:	1c7d                	addi	s8,s8,-1
1c00bf42:	4505                	li	a0,1
1c00bf44:	bfe9                	j	1c00bf1e <_prf+0x528>
1c00bf46:	06600593          	li	a1,102
1c00bf4a:	4c01                	li	s8,0
1c00bf4c:	f4ba9de3          	bne	s5,a1,1c00bea6 <_prf+0x4b0>
1c00bf50:	01ab85b3          	add	a1,s7,s10
1c00bf54:	06600a93          	li	s5,102
1c00bf58:	f405d9e3          	bgez	a1,1c00beaa <_prf+0x4b4>
1c00bf5c:	45c1                	li	a1,16
1c00bf5e:	d62e                	sw	a1,44(sp)
1c00bf60:	4d81                	li	s11,0
1c00bf62:	4301                	li	t1,0
1c00bf64:	080003b7          	lui	t2,0x8000
1c00bf68:	dc1a                	sw	t1,56(sp)
1c00bf6a:	de1e                	sw	t2,60(sp)
1c00bf6c:	1dfd                	addi	s11,s11,-1
1c00bf6e:	09fdb363          	p.bneimm	s11,-1,1c00bff4 <_prf+0x5fe>
1c00bf72:	55c2                	lw	a1,48(sp)
1c00bf74:	5562                	lw	a0,56(sp)
1c00bf76:	58d2                	lw	a7,52(sp)
1c00bf78:	5372                	lw	t1,60(sp)
1c00bf7a:	952e                	add	a0,a0,a1
1c00bf7c:	00b535b3          	sltu	a1,a0,a1
1c00bf80:	989a                	add	a7,a7,t1
1c00bf82:	95c6                	add	a1,a1,a7
1c00bf84:	da2e                	sw	a1,52(sp)
1c00bf86:	d82a                	sw	a0,48(sp)
1c00bf88:	f605b5b3          	p.bclr	a1,a1,27,0
1c00bf8c:	c981                	beqz	a1,1c00bf9c <_prf+0x5a6>
1c00bf8e:	1808                	addi	a0,sp,48
1c00bf90:	9edff0ef          	jal	ra,1c00b97c <_ldiv5>
1c00bf94:	1808                	addi	a0,sp,48
1c00bf96:	9c7ff0ef          	jal	ra,1c00b95c <_rlrshift>
1c00bf9a:	0b85                	addi	s7,s7,1
1c00bf9c:	06600593          	li	a1,102
1c00bfa0:	001b0d93          	addi	s11,s6,1
1c00bfa4:	08ba9463          	bne	s5,a1,1c00c02c <_prf+0x636>
1c00bfa8:	05705d63          	blez	s7,1c00c002 <_prf+0x60c>
1c00bfac:	017b0db3          	add	s11,s6,s7
1c00bfb0:	106c                	addi	a1,sp,44
1c00bfb2:	1808                	addi	a0,sp,48
1c00bfb4:	a0dff0ef          	jal	ra,1c00b9c0 <_get_digit>
1c00bfb8:	00ab00ab          	p.sb	a0,1(s6!)
1c00bfbc:	ffbb1ae3          	bne	s6,s11,1c00bfb0 <_prf+0x5ba>
1c00bfc0:	4b81                	li	s7,0
1c00bfc2:	000c9463          	bnez	s9,1c00bfca <_prf+0x5d4>
1c00bfc6:	020d0163          	beqz	s10,1c00bfe8 <_prf+0x5f2>
1c00bfca:	001d8b13          	addi	s6,s11,1
1c00bfce:	02e00613          	li	a2,46
1c00bfd2:	00cd8023          	sb	a2,0(s11)
1c00bfd6:	8cea                	mv	s9,s10
1c00bfd8:	8dda                	mv	s11,s6
1c00bfda:	03000893          	li	a7,48
1c00bfde:	1cfd                	addi	s9,s9,-1
1c00bfe0:	03fcb663          	p.bneimm	s9,-1,1c00c00c <_prf+0x616>
1c00bfe4:	01ab0db3          	add	s11,s6,s10
1c00bfe8:	060c1c63          	bnez	s8,1c00c060 <_prf+0x66a>
1c00bfec:	a8c1                	j	1c00c0bc <_prf+0x6c6>
1c00bfee:	06500a93          	li	s5,101
1c00bff2:	bd55                	j	1c00bea6 <_prf+0x4b0>
1c00bff4:	1828                	addi	a0,sp,56
1c00bff6:	987ff0ef          	jal	ra,1c00b97c <_ldiv5>
1c00bffa:	1828                	addi	a0,sp,56
1c00bffc:	961ff0ef          	jal	ra,1c00b95c <_rlrshift>
1c00c000:	b7b5                	j	1c00bf6c <_prf+0x576>
1c00c002:	03000593          	li	a1,48
1c00c006:	00bb0023          	sb	a1,0(s6)
1c00c00a:	bf65                	j	1c00bfc2 <_prf+0x5cc>
1c00c00c:	0d85                	addi	s11,s11,1
1c00c00e:	000b8663          	beqz	s7,1c00c01a <_prf+0x624>
1c00c012:	ff1d8fa3          	sb	a7,-1(s11)
1c00c016:	0b85                	addi	s7,s7,1
1c00c018:	b7d9                	j	1c00bfde <_prf+0x5e8>
1c00c01a:	106c                	addi	a1,sp,44
1c00c01c:	1808                	addi	a0,sp,48
1c00c01e:	c446                	sw	a7,8(sp)
1c00c020:	9a1ff0ef          	jal	ra,1c00b9c0 <_get_digit>
1c00c024:	fead8fa3          	sb	a0,-1(s11)
1c00c028:	48a2                	lw	a7,8(sp)
1c00c02a:	bf55                	j	1c00bfde <_prf+0x5e8>
1c00c02c:	106c                	addi	a1,sp,44
1c00c02e:	1808                	addi	a0,sp,48
1c00c030:	991ff0ef          	jal	ra,1c00b9c0 <_get_digit>
1c00c034:	00ab0023          	sb	a0,0(s6)
1c00c038:	03000593          	li	a1,48
1c00c03c:	00b50363          	beq	a0,a1,1c00c042 <_prf+0x64c>
1c00c040:	1bfd                	addi	s7,s7,-1
1c00c042:	000c9463          	bnez	s9,1c00c04a <_prf+0x654>
1c00c046:	000d0b63          	beqz	s10,1c00c05c <_prf+0x666>
1c00c04a:	002b0d93          	addi	s11,s6,2
1c00c04e:	02e00593          	li	a1,46
1c00c052:	00bb00a3          	sb	a1,1(s6)
1c00c056:	9d6e                	add	s10,s10,s11
1c00c058:	07bd1863          	bne	s10,s11,1c00c0c8 <_prf+0x6d2>
1c00c05c:	000c0f63          	beqz	s8,1c00c07a <_prf+0x684>
1c00c060:	03000593          	li	a1,48
1c00c064:	fffd8713          	addi	a4,s11,-1
1c00c068:	00074603          	lbu	a2,0(a4) # 80000000 <pulp__FC+0x80000001>
1c00c06c:	06b60563          	beq	a2,a1,1c00c0d6 <_prf+0x6e0>
1c00c070:	02e00593          	li	a1,46
1c00c074:	00b61363          	bne	a2,a1,1c00c07a <_prf+0x684>
1c00c078:	8dba                	mv	s11,a4
1c00c07a:	c05ab733          	p.bclr	a4,s5,0,5
1c00c07e:	04500613          	li	a2,69
1c00c082:	02c71d63          	bne	a4,a2,1c00c0bc <_prf+0x6c6>
1c00c086:	87d6                	mv	a5,s5
1c00c088:	00fd8023          	sb	a5,0(s11)
1c00c08c:	02b00793          	li	a5,43
1c00c090:	000bd663          	bgez	s7,1c00c09c <_prf+0x6a6>
1c00c094:	41700bb3          	neg	s7,s7
1c00c098:	02d00793          	li	a5,45
1c00c09c:	00fd80a3          	sb	a5,1(s11)
1c00c0a0:	47a9                	li	a5,10
1c00c0a2:	02fbc733          	div	a4,s7,a5
1c00c0a6:	0d91                	addi	s11,s11,4
1c00c0a8:	02fbe6b3          	rem	a3,s7,a5
1c00c0ac:	03070713          	addi	a4,a4,48
1c00c0b0:	feed8f23          	sb	a4,-2(s11)
1c00c0b4:	03068693          	addi	a3,a3,48
1c00c0b8:	fedd8fa3          	sb	a3,-1(s11)
1c00c0bc:	00c8                	addi	a0,sp,68
1c00c0be:	000d8023          	sb	zero,0(s11)
1c00c0c2:	40ad8533          	sub	a0,s11,a0
1c00c0c6:	b9e5                	j	1c00bdbe <_prf+0x3c8>
1c00c0c8:	106c                	addi	a1,sp,44
1c00c0ca:	1808                	addi	a0,sp,48
1c00c0cc:	8f5ff0ef          	jal	ra,1c00b9c0 <_get_digit>
1c00c0d0:	00ad80ab          	p.sb	a0,1(s11!)
1c00c0d4:	b751                	j	1c00c058 <_prf+0x662>
1c00c0d6:	8dba                	mv	s11,a4
1c00c0d8:	b771                	j	1c00c064 <_prf+0x66e>
1c00c0da:	000c2783          	lw	a5,0(s8)
1c00c0de:	004c0a13          	addi	s4,s8,4
1c00c0e2:	0127a023          	sw	s2,0(a5) # 80000000 <pulp__FC+0x80000001>
1c00c0e6:	b27d                	j	1c00ba94 <_prf+0x9e>
1c00c0e8:	004c0a13          	addi	s4,s8,4
1c00c0ec:	000c2583          	lw	a1,0(s8)
1c00c0f0:	00dc                	addi	a5,sp,68
1c00c0f2:	040c8263          	beqz	s9,1c00c136 <_prf+0x740>
1c00c0f6:	03000693          	li	a3,48
1c00c0fa:	04d10223          	sb	a3,68(sp)
1c00c0fe:	04510513          	addi	a0,sp,69
1c00c102:	e99d                	bnez	a1,1c00c138 <_prf+0x742>
1c00c104:	040102a3          	sb	zero,69(sp)
1c00c108:	4401                	li	s0,0
1c00c10a:	0dfd3063          	p.bneimm	s10,-1,1c00c1ca <_prf+0x7d4>
1c00c10e:	04410b93          	addi	s7,sp,68
1c00c112:	0d3cc063          	blt	s9,s3,1c00c1d2 <_prf+0x7dc>
1c00c116:	89e6                	mv	s3,s9
1c00c118:	41790433          	sub	s0,s2,s7
1c00c11c:	01740933          	add	s2,s0,s7
1c00c120:	96098ae3          	beqz	s3,1c00ba94 <_prf+0x9e>
1c00c124:	45f2                	lw	a1,28(sp)
1c00c126:	001bc50b          	p.lbu	a0,1(s7!)
1c00c12a:	47e2                	lw	a5,24(sp)
1c00c12c:	9782                	jalr	a5
1c00c12e:	93f522e3          	p.beqimm	a0,-1,1c00ba52 <_prf+0x5c>
1c00c132:	19fd                	addi	s3,s3,-1
1c00c134:	b7e5                	j	1c00c11c <_prf+0x726>
1c00c136:	853e                	mv	a0,a5
1c00c138:	86ea                	mv	a3,s10
1c00c13a:	4621                	li	a2,8
1c00c13c:	40f50433          	sub	s0,a0,a5
1c00c140:	fc6ff0ef          	jal	ra,1c00b906 <_to_x>
1c00c144:	9522                	add	a0,a0,s0
1c00c146:	4401                	li	s0,0
1c00c148:	c9fd25e3          	p.beqimm	s10,-1,1c00bdd2 <_prf+0x3dc>
1c00c14c:	02000793          	li	a5,32
1c00c150:	c63e                	sw	a5,12(sp)
1c00c152:	b141                	j	1c00bdd2 <_prf+0x3dc>
1c00c154:	000c2583          	lw	a1,0(s8)
1c00c158:	77e1                	lui	a5,0xffff8
1c00c15a:	8307c793          	xori	a5,a5,-2000
1c00c15e:	46a1                	li	a3,8
1c00c160:	4641                	li	a2,16
1c00c162:	04610513          	addi	a0,sp,70
1c00c166:	04f11223          	sh	a5,68(sp)
1c00c16a:	f9cff0ef          	jal	ra,1c00b906 <_to_x>
1c00c16e:	004c0a13          	addi	s4,s8,4
1c00c172:	0509                	addi	a0,a0,2
1c00c174:	4401                	li	s0,0
1c00c176:	be5d                	j	1c00bd2c <_prf+0x336>
1c00c178:	000d4463          	bltz	s10,1c00c180 <_prf+0x78a>
1c00c17c:	05acccb3          	p.min	s9,s9,s10
1c00c180:	900c8ae3          	beqz	s9,1c00ba94 <_prf+0x9e>
1c00c184:	8666                	mv	a2,s9
1c00c186:	00c8                	addi	a0,sp,68
1c00c188:	d0aff0ef          	jal	ra,1c00b692 <memcpy>
1c00c18c:	b6ad                	j	1c00bcf6 <_prf+0x300>
1c00c18e:	000c2583          	lw	a1,0(s8)
1c00c192:	86ea                	mv	a3,s10
1c00c194:	4629                	li	a2,10
1c00c196:	00c8                	addi	a0,sp,68
1c00c198:	004c0a13          	addi	s4,s8,4
1c00c19c:	f6aff0ef          	jal	ra,1c00b906 <_to_x>
1c00c1a0:	b75d                	j	1c00c146 <_prf+0x750>
1c00c1a2:	f9f78613          	addi	a2,a5,-97 # ffff7f9f <pulp__FC+0xffff7fa0>
1c00c1a6:	0ff67613          	andi	a2,a2,255
1c00c1aa:	aac5e1e3          	bltu	a1,a2,1c00bc4c <_prf+0x256>
1c00c1ae:	1781                	addi	a5,a5,-32
1c00c1b0:	fef68fa3          	sb	a5,-1(a3)
1c00c1b4:	bc61                	j	1c00bc4c <_prf+0x256>
1c00c1b6:	45f2                	lw	a1,28(sp)
1c00c1b8:	4762                	lw	a4,24(sp)
1c00c1ba:	02500513          	li	a0,37
1c00c1be:	9702                	jalr	a4
1c00c1c0:	89f529e3          	p.beqimm	a0,-1,1c00ba52 <_prf+0x5c>
1c00c1c4:	0905                	addi	s2,s2,1
1c00c1c6:	8a62                	mv	s4,s8
1c00c1c8:	b0f1                	j	1c00ba94 <_prf+0x9e>
1c00c1ca:	02000793          	li	a5,32
1c00c1ce:	c63e                	sw	a5,12(sp)
1c00c1d0:	bf3d                	j	1c00c10e <_prf+0x718>
1c00c1d2:	4752                	lw	a4,20(sp)
1c00c1d4:	cf01                	beqz	a4,1c00c1ec <_prf+0x7f6>
1c00c1d6:	019b8833          	add	a6,s7,s9
1c00c1da:	02000713          	li	a4,32
1c00c1de:	417807b3          	sub	a5,a6,s7
1c00c1e2:	f337dbe3          	ble	s3,a5,1c00c118 <_prf+0x722>
1c00c1e6:	00e800ab          	p.sb	a4,1(a6!)
1c00c1ea:	bfd5                	j	1c00c1de <_prf+0x7e8>
1c00c1ec:	41998c33          	sub	s8,s3,s9
1c00c1f0:	001c8613          	addi	a2,s9,1
1c00c1f4:	85de                	mv	a1,s7
1c00c1f6:	018b8533          	add	a0,s7,s8
1c00c1fa:	cceff0ef          	jal	ra,1c00b6c8 <memmove>
1c00c1fe:	4732                	lw	a4,12(sp)
1c00c200:	02000793          	li	a5,32
1c00c204:	00f70363          	beq	a4,a5,1c00c20a <_prf+0x814>
1c00c208:	ca22                	sw	s0,20(sp)
1c00c20a:	47d2                	lw	a5,20(sp)
1c00c20c:	9c3e                	add	s8,s8,a5
1c00c20e:	00fb8ab3          	add	s5,s7,a5
1c00c212:	417a87b3          	sub	a5,s5,s7
1c00c216:	f187d1e3          	ble	s8,a5,1c00c118 <_prf+0x722>
1c00c21a:	4732                	lw	a4,12(sp)
1c00c21c:	00ea80ab          	p.sb	a4,1(s5!)
1c00c220:	bfcd                	j	1c00c212 <_prf+0x81c>
1c00c222:	4442                	lw	s0,16(sp)
1c00c224:	b67d                	j	1c00bdd2 <_prf+0x3dc>

1c00c226 <__rt_uart_cluster_req_done>:
1c00c226:	300476f3          	csrrci	a3,mstatus,8
1c00c22a:	4785                	li	a5,1
1c00c22c:	08f50c23          	sb	a5,152(a0)
1c00c230:	09954783          	lbu	a5,153(a0)
1c00c234:	00201737          	lui	a4,0x201
1c00c238:	e0470713          	addi	a4,a4,-508 # 200e04 <__l1_heap_size+0x1e4e1c>
1c00c23c:	04078793          	addi	a5,a5,64
1c00c240:	07da                	slli	a5,a5,0x16
1c00c242:	0007e723          	p.sw	zero,a4(a5)
1c00c246:	30069073          	csrw	mstatus,a3
1c00c24a:	8082                	ret

1c00c24c <__rt_uart_cluster_req>:
1c00c24c:	1141                	addi	sp,sp,-16
1c00c24e:	c606                	sw	ra,12(sp)
1c00c250:	c422                	sw	s0,8(sp)
1c00c252:	30047473          	csrrci	s0,mstatus,8
1c00c256:	1c00c7b7          	lui	a5,0x1c00c
1c00c25a:	22678793          	addi	a5,a5,550 # 1c00c226 <__rt_uart_cluster_req_done>
1c00c25e:	c55c                	sw	a5,12(a0)
1c00c260:	4785                	li	a5,1
1c00c262:	d55c                	sw	a5,44(a0)
1c00c264:	411c                	lw	a5,0(a0)
1c00c266:	02052823          	sw	zero,48(a0)
1c00c26a:	c908                	sw	a0,16(a0)
1c00c26c:	43cc                	lw	a1,4(a5)
1c00c26e:	4514                	lw	a3,8(a0)
1c00c270:	4150                	lw	a2,4(a0)
1c00c272:	0586                	slli	a1,a1,0x1
1c00c274:	00c50793          	addi	a5,a0,12
1c00c278:	4701                	li	a4,0
1c00c27a:	0585                	addi	a1,a1,1
1c00c27c:	4501                	li	a0,0
1c00c27e:	f07fe0ef          	jal	ra,1c00b184 <rt_periph_copy>
1c00c282:	30041073          	csrw	mstatus,s0
1c00c286:	40b2                	lw	ra,12(sp)
1c00c288:	4422                	lw	s0,8(sp)
1c00c28a:	0141                	addi	sp,sp,16
1c00c28c:	8082                	ret

1c00c28e <__rt_uart_wait_tx_done.isra.2>:
1c00c28e:	1a1026b7          	lui	a3,0x1a102
1c00c292:	09068693          	addi	a3,a3,144 # 1a102090 <__l1_end+0xa0fe078>
1c00c296:	411c                	lw	a5,0(a0)
1c00c298:	079e                	slli	a5,a5,0x7
1c00c29a:	00d78733          	add	a4,a5,a3
1c00c29e:	0721                	addi	a4,a4,8
1c00c2a0:	4318                	lw	a4,0(a4)
1c00c2a2:	8b41                	andi	a4,a4,16
1c00c2a4:	ef39                	bnez	a4,1c00c302 <__rt_uart_wait_tx_done.isra.2+0x74>
1c00c2a6:	1a102737          	lui	a4,0x1a102
1c00c2aa:	0a070713          	addi	a4,a4,160 # 1a1020a0 <__l1_end+0xa0fe088>
1c00c2ae:	97ba                	add	a5,a5,a4
1c00c2b0:	4398                	lw	a4,0(a5)
1c00c2b2:	fc173733          	p.bclr	a4,a4,30,1
1c00c2b6:	ff6d                	bnez	a4,1c00c2b0 <__rt_uart_wait_tx_done.isra.2+0x22>
1c00c2b8:	f14027f3          	csrr	a5,mhartid
1c00c2bc:	8795                	srai	a5,a5,0x5
1c00c2be:	1a109737          	lui	a4,0x1a109
1c00c2c2:	00204637          	lui	a2,0x204
1c00c2c6:	f267b7b3          	p.bclr	a5,a5,25,6
1c00c2ca:	01470813          	addi	a6,a4,20 # 1a109014 <__l1_end+0xa104ffc>
1c00c2ce:	01460e13          	addi	t3,a2,20 # 204014 <__l1_heap_size+0x1e802c>
1c00c2d2:	00470e93          	addi	t4,a4,4
1c00c2d6:	6691                	lui	a3,0x4
1c00c2d8:	6311                	lui	t1,0x4
1c00c2da:	457d                	li	a0,31
1c00c2dc:	88be                	mv	a7,a5
1c00c2de:	0641                	addi	a2,a2,16
1c00c2e0:	0721                	addi	a4,a4,8
1c00c2e2:	03200593          	li	a1,50
1c00c2e6:	00682023          	sw	t1,0(a6)
1c00c2ea:	00a79f63          	bne	a5,a0,1c00c308 <__rt_uart_wait_tx_done.isra.2+0x7a>
1c00c2ee:	00dea023          	sw	a3,0(t4)
1c00c2f2:	10500073          	wfi
1c00c2f6:	00a89c63          	bne	a7,a0,1c00c30e <__rt_uart_wait_tx_done.isra.2+0x80>
1c00c2fa:	c314                	sw	a3,0(a4)
1c00c2fc:	15fd                	addi	a1,a1,-1
1c00c2fe:	f5e5                	bnez	a1,1c00c2e6 <__rt_uart_wait_tx_done.isra.2+0x58>
1c00c300:	8082                	ret
1c00c302:	10500073          	wfi
1c00c306:	bf41                	j	1c00c296 <__rt_uart_wait_tx_done.isra.2+0x8>
1c00c308:	00de2023          	sw	a3,0(t3)
1c00c30c:	b7dd                	j	1c00c2f2 <__rt_uart_wait_tx_done.isra.2+0x64>
1c00c30e:	c214                	sw	a3,0(a2)
1c00c310:	b7f5                	j	1c00c2fc <__rt_uart_wait_tx_done.isra.2+0x6e>

1c00c312 <__rt_uart_setup>:
1c00c312:	4518                	lw	a4,8(a0)
1c00c314:	1c0016b7          	lui	a3,0x1c001
1c00c318:	7e86a683          	lw	a3,2024(a3) # 1c0017e8 <__rt_freq_domains>
1c00c31c:	00175793          	srli	a5,a4,0x1
1c00c320:	97b6                	add	a5,a5,a3
1c00c322:	02e7d7b3          	divu	a5,a5,a4
1c00c326:	4154                	lw	a3,4(a0)
1c00c328:	1a102737          	lui	a4,0x1a102
1c00c32c:	0a470713          	addi	a4,a4,164 # 1a1020a4 <__l1_end+0xa0fe08c>
1c00c330:	069e                	slli	a3,a3,0x7
1c00c332:	17fd                	addi	a5,a5,-1
1c00c334:	07c2                	slli	a5,a5,0x10
1c00c336:	3067e793          	ori	a5,a5,774
1c00c33a:	00f6e723          	p.sw	a5,a4(a3)
1c00c33e:	8082                	ret

1c00c340 <__rt_uart_setfreq_after>:
1c00c340:	1c001537          	lui	a0,0x1c001
1c00c344:	70852783          	lw	a5,1800(a0) # 1c001708 <__rt_uart>
1c00c348:	1141                	addi	sp,sp,-16
1c00c34a:	c422                	sw	s0,8(sp)
1c00c34c:	c606                	sw	ra,12(sp)
1c00c34e:	70850413          	addi	s0,a0,1800
1c00c352:	c781                	beqz	a5,1c00c35a <__rt_uart_setfreq_after+0x1a>
1c00c354:	70850513          	addi	a0,a0,1800
1c00c358:	3f6d                	jal	1c00c312 <__rt_uart_setup>
1c00c35a:	481c                	lw	a5,16(s0)
1c00c35c:	c781                	beqz	a5,1c00c364 <__rt_uart_setfreq_after+0x24>
1c00c35e:	01040513          	addi	a0,s0,16
1c00c362:	3f45                	jal	1c00c312 <__rt_uart_setup>
1c00c364:	501c                	lw	a5,32(s0)
1c00c366:	c781                	beqz	a5,1c00c36e <__rt_uart_setfreq_after+0x2e>
1c00c368:	02040513          	addi	a0,s0,32
1c00c36c:	375d                	jal	1c00c312 <__rt_uart_setup>
1c00c36e:	40b2                	lw	ra,12(sp)
1c00c370:	4422                	lw	s0,8(sp)
1c00c372:	4501                	li	a0,0
1c00c374:	0141                	addi	sp,sp,16
1c00c376:	8082                	ret

1c00c378 <soc_eu_fcEventMask_setEvent>:
1c00c378:	02000793          	li	a5,32
1c00c37c:	02f54733          	div	a4,a0,a5
1c00c380:	1a1066b7          	lui	a3,0x1a106
1c00c384:	0691                	addi	a3,a3,4
1c00c386:	02f56533          	rem	a0,a0,a5
1c00c38a:	070a                	slli	a4,a4,0x2
1c00c38c:	9736                	add	a4,a4,a3
1c00c38e:	4314                	lw	a3,0(a4)
1c00c390:	4785                	li	a5,1
1c00c392:	00a797b3          	sll	a5,a5,a0
1c00c396:	fff7c793          	not	a5,a5
1c00c39a:	8ff5                	and	a5,a5,a3
1c00c39c:	c31c                	sw	a5,0(a4)
1c00c39e:	8082                	ret

1c00c3a0 <__rt_uart_setfreq_before>:
1c00c3a0:	1101                	addi	sp,sp,-32
1c00c3a2:	cc22                	sw	s0,24(sp)
1c00c3a4:	c84a                	sw	s2,16(sp)
1c00c3a6:	c64e                	sw	s3,12(sp)
1c00c3a8:	1c001437          	lui	s0,0x1c001
1c00c3ac:	1a102937          	lui	s2,0x1a102
1c00c3b0:	005009b7          	lui	s3,0x500
1c00c3b4:	ca26                	sw	s1,20(sp)
1c00c3b6:	ce06                	sw	ra,28(sp)
1c00c3b8:	70840413          	addi	s0,s0,1800 # 1c001708 <__rt_uart>
1c00c3bc:	4481                	li	s1,0
1c00c3be:	0a490913          	addi	s2,s2,164 # 1a1020a4 <__l1_end+0xa0fe08c>
1c00c3c2:	0999                	addi	s3,s3,6
1c00c3c4:	401c                	lw	a5,0(s0)
1c00c3c6:	cb81                	beqz	a5,1c00c3d6 <__rt_uart_setfreq_before+0x36>
1c00c3c8:	00440513          	addi	a0,s0,4
1c00c3cc:	35c9                	jal	1c00c28e <__rt_uart_wait_tx_done.isra.2>
1c00c3ce:	405c                	lw	a5,4(s0)
1c00c3d0:	079e                	slli	a5,a5,0x7
1c00c3d2:	0137e923          	p.sw	s3,s2(a5)
1c00c3d6:	0485                	addi	s1,s1,1
1c00c3d8:	0441                	addi	s0,s0,16
1c00c3da:	fe34b5e3          	p.bneimm	s1,3,1c00c3c4 <__rt_uart_setfreq_before+0x24>
1c00c3de:	40f2                	lw	ra,28(sp)
1c00c3e0:	4462                	lw	s0,24(sp)
1c00c3e2:	44d2                	lw	s1,20(sp)
1c00c3e4:	4942                	lw	s2,16(sp)
1c00c3e6:	49b2                	lw	s3,12(sp)
1c00c3e8:	4501                	li	a0,0
1c00c3ea:	6105                	addi	sp,sp,32
1c00c3ec:	8082                	ret

1c00c3ee <rt_uart_conf_init>:
1c00c3ee:	000997b7          	lui	a5,0x99
1c00c3f2:	96878793          	addi	a5,a5,-1688 # 98968 <__l1_heap_size+0x7c980>
1c00c3f6:	c11c                	sw	a5,0(a0)
1c00c3f8:	57fd                	li	a5,-1
1c00c3fa:	c15c                	sw	a5,4(a0)
1c00c3fc:	8082                	ret

1c00c3fe <__rt_uart_open>:
1c00c3fe:	1141                	addi	sp,sp,-16
1c00c400:	c606                	sw	ra,12(sp)
1c00c402:	c422                	sw	s0,8(sp)
1c00c404:	c226                	sw	s1,4(sp)
1c00c406:	c04a                	sw	s2,0(sp)
1c00c408:	30047973          	csrrci	s2,mstatus,8
1c00c40c:	cd8d                	beqz	a1,1c00c446 <__rt_uart_open+0x48>
1c00c40e:	4198                	lw	a4,0(a1)
1c00c410:	1c0016b7          	lui	a3,0x1c001
1c00c414:	ffc50793          	addi	a5,a0,-4
1c00c418:	70868413          	addi	s0,a3,1800 # 1c001708 <__rt_uart>
1c00c41c:	0792                	slli	a5,a5,0x4
1c00c41e:	943e                	add	s0,s0,a5
1c00c420:	4010                	lw	a2,0(s0)
1c00c422:	70868693          	addi	a3,a3,1800
1c00c426:	c60d                	beqz	a2,1c00c450 <__rt_uart_open+0x52>
1c00c428:	c589                	beqz	a1,1c00c432 <__rt_uart_open+0x34>
1c00c42a:	418c                	lw	a1,0(a1)
1c00c42c:	4418                	lw	a4,8(s0)
1c00c42e:	04e59863          	bne	a1,a4,1c00c47e <__rt_uart_open+0x80>
1c00c432:	0605                	addi	a2,a2,1
1c00c434:	00c6e7a3          	p.sw	a2,a5(a3)
1c00c438:	8522                	mv	a0,s0
1c00c43a:	40b2                	lw	ra,12(sp)
1c00c43c:	4422                	lw	s0,8(sp)
1c00c43e:	4492                	lw	s1,4(sp)
1c00c440:	4902                	lw	s2,0(sp)
1c00c442:	0141                	addi	sp,sp,16
1c00c444:	8082                	ret
1c00c446:	00099737          	lui	a4,0x99
1c00c44a:	96870713          	addi	a4,a4,-1688 # 98968 <__l1_heap_size+0x7c980>
1c00c44e:	b7c9                	j	1c00c410 <__rt_uart_open+0x12>
1c00c450:	4785                	li	a5,1
1c00c452:	c01c                	sw	a5,0(s0)
1c00c454:	c418                	sw	a4,8(s0)
1c00c456:	c048                	sw	a0,4(s0)
1c00c458:	1a102737          	lui	a4,0x1a102
1c00c45c:	4314                	lw	a3,0(a4)
1c00c45e:	00a797b3          	sll	a5,a5,a0
1c00c462:	00251493          	slli	s1,a0,0x2
1c00c466:	8fd5                	or	a5,a5,a3
1c00c468:	c31c                	sw	a5,0(a4)
1c00c46a:	8526                	mv	a0,s1
1c00c46c:	3731                	jal	1c00c378 <soc_eu_fcEventMask_setEvent>
1c00c46e:	00148513          	addi	a0,s1,1
1c00c472:	3719                	jal	1c00c378 <soc_eu_fcEventMask_setEvent>
1c00c474:	8522                	mv	a0,s0
1c00c476:	3d71                	jal	1c00c312 <__rt_uart_setup>
1c00c478:	30091073          	csrw	mstatus,s2
1c00c47c:	bf75                	j	1c00c438 <__rt_uart_open+0x3a>
1c00c47e:	4401                	li	s0,0
1c00c480:	bf65                	j	1c00c438 <__rt_uart_open+0x3a>

1c00c482 <rt_uart_close>:
1c00c482:	1141                	addi	sp,sp,-16
1c00c484:	c606                	sw	ra,12(sp)
1c00c486:	c422                	sw	s0,8(sp)
1c00c488:	c226                	sw	s1,4(sp)
1c00c48a:	300474f3          	csrrci	s1,mstatus,8
1c00c48e:	411c                	lw	a5,0(a0)
1c00c490:	17fd                	addi	a5,a5,-1
1c00c492:	c11c                	sw	a5,0(a0)
1c00c494:	eb85                	bnez	a5,1c00c4c4 <rt_uart_close+0x42>
1c00c496:	842a                	mv	s0,a0
1c00c498:	0511                	addi	a0,a0,4
1c00c49a:	3bd5                	jal	1c00c28e <__rt_uart_wait_tx_done.isra.2>
1c00c49c:	405c                	lw	a5,4(s0)
1c00c49e:	1a102737          	lui	a4,0x1a102
1c00c4a2:	00500637          	lui	a2,0x500
1c00c4a6:	079e                	slli	a5,a5,0x7
1c00c4a8:	0a470693          	addi	a3,a4,164 # 1a1020a4 <__l1_end+0xa0fe08c>
1c00c4ac:	0619                	addi	a2,a2,6
1c00c4ae:	00c7e6a3          	p.sw	a2,a3(a5)
1c00c4b2:	4050                	lw	a2,4(s0)
1c00c4b4:	4314                	lw	a3,0(a4)
1c00c4b6:	4785                	li	a5,1
1c00c4b8:	00c797b3          	sll	a5,a5,a2
1c00c4bc:	fff7c793          	not	a5,a5
1c00c4c0:	8ff5                	and	a5,a5,a3
1c00c4c2:	c31c                	sw	a5,0(a4)
1c00c4c4:	30049073          	csrw	mstatus,s1
1c00c4c8:	40b2                	lw	ra,12(sp)
1c00c4ca:	4422                	lw	s0,8(sp)
1c00c4cc:	4492                	lw	s1,4(sp)
1c00c4ce:	0141                	addi	sp,sp,16
1c00c4d0:	8082                	ret

1c00c4d2 <rt_uart_cluster_write>:
1c00c4d2:	f14027f3          	csrr	a5,mhartid
1c00c4d6:	8795                	srai	a5,a5,0x5
1c00c4d8:	f267b7b3          	p.bclr	a5,a5,25,6
1c00c4dc:	08f68ca3          	sb	a5,153(a3)
1c00c4e0:	1c00c7b7          	lui	a5,0x1c00c
1c00c4e4:	24c78793          	addi	a5,a5,588 # 1c00c24c <__rt_uart_cluster_req>
1c00c4e8:	c6dc                	sw	a5,12(a3)
1c00c4ea:	4785                	li	a5,1
1c00c4ec:	c288                	sw	a0,0(a3)
1c00c4ee:	c2cc                	sw	a1,4(a3)
1c00c4f0:	c690                	sw	a2,8(a3)
1c00c4f2:	08068c23          	sb	zero,152(a3)
1c00c4f6:	0206a823          	sw	zero,48(a3)
1c00c4fa:	ca94                	sw	a3,16(a3)
1c00c4fc:	d6dc                	sw	a5,44(a3)
1c00c4fe:	00c68513          	addi	a0,a3,12
1c00c502:	dadfd06f          	j	1c00a2ae <__rt_cluster_push_fc_event>

1c00c506 <__rt_uart_init>:
1c00c506:	1c00c5b7          	lui	a1,0x1c00c
1c00c50a:	1141                	addi	sp,sp,-16
1c00c50c:	4601                	li	a2,0
1c00c50e:	3a058593          	addi	a1,a1,928 # 1c00c3a0 <__rt_uart_setfreq_before>
1c00c512:	4511                	li	a0,4
1c00c514:	c606                	sw	ra,12(sp)
1c00c516:	c422                	sw	s0,8(sp)
1c00c518:	dc8fe0ef          	jal	ra,1c00aae0 <__rt_cbsys_add>
1c00c51c:	1c00c5b7          	lui	a1,0x1c00c
1c00c520:	842a                	mv	s0,a0
1c00c522:	4601                	li	a2,0
1c00c524:	34058593          	addi	a1,a1,832 # 1c00c340 <__rt_uart_setfreq_after>
1c00c528:	4515                	li	a0,5
1c00c52a:	db6fe0ef          	jal	ra,1c00aae0 <__rt_cbsys_add>
1c00c52e:	1c0017b7          	lui	a5,0x1c001
1c00c532:	70878793          	addi	a5,a5,1800 # 1c001708 <__rt_uart>
1c00c536:	0007a023          	sw	zero,0(a5)
1c00c53a:	0007a823          	sw	zero,16(a5)
1c00c53e:	0207a023          	sw	zero,32(a5)
1c00c542:	8d41                	or	a0,a0,s0
1c00c544:	c10d                	beqz	a0,1c00c566 <__rt_uart_init+0x60>
1c00c546:	f1402673          	csrr	a2,mhartid
1c00c54a:	1c001537          	lui	a0,0x1c001
1c00c54e:	40565593          	srai	a1,a2,0x5
1c00c552:	f265b5b3          	p.bclr	a1,a1,25,6
1c00c556:	f4563633          	p.bclr	a2,a2,26,5
1c00c55a:	ba050513          	addi	a0,a0,-1120 # 1c000ba0 <PIo2+0x244>
1c00c55e:	b7eff0ef          	jal	ra,1c00b8dc <printf>
1c00c562:	b08ff0ef          	jal	ra,1c00b86a <abort>
1c00c566:	40b2                	lw	ra,12(sp)
1c00c568:	4422                	lw	s0,8(sp)
1c00c56a:	0141                	addi	sp,sp,16
1c00c56c:	8082                	ret

1c00c56e <_endtext>:
	...

Disassembly of section .l2_data:

1c010000 <__cluster_text_start>:
1c010000:	f1402573          	csrr	a0,mhartid
1c010004:	01f57593          	andi	a1,a0,31
1c010008:	8115                	srli	a0,a0,0x5
1c01000a:	000702b7          	lui	t0,0x70
1c01000e:	00204337          	lui	t1,0x204
1c010012:	00532023          	sw	t0,0(t1) # 204000 <__l1_heap_size+0x1e8018>
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
1c01004a:	010c8c93          	addi	s9,s9,16 # 1a109010 <__l1_end+0xa104ff8>
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
1c0100ce:	09e9a223          	sw	t5,132(s3) # 204084 <__l1_heap_size+0x1e809c>
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
1c010136:	23c96283          	p.elw	t0,572(s2) # 20423c <__l1_heap_size+0x1e8254>

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
