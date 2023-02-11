version 1.0

task make_mask_and_diff {
	# This combines the creation of the bed graph histogram mask file and the
	# creation of the diff file into one WDL task. Sometimes, in WDL, it is
	# easier to combine tasks to avoid shenanigans with the Array[File] type.
	# The masking part of this task is based on github.com/aofarrel/mask-by-coverage
	input {
		File bam
		File vcf
		File tbmf
		Int min_coverage
		Boolean histograms = false

		# runtime attributes
		Int addldisk = 250
		Int cpu      = 16
		Int retries  = 1
		Int memory   = 32
		Int preempt  = 1
	}
	String basename_bam = basename(bam, ".bam")
	Int finalDiskSize = ceil(size(bam, "GB")) + addldisk
	
	command <<<
	set -eux pipefail
	cp ~{bam} .
	samtools sort -u ~{basename_bam}.bam > sorted_u_~{basename_bam}.bam
	bedtools genomecov -ibam sorted_u_~{basename_bam}.bam -bga | \
		awk '$4 < ~{min_coverage}' > \
		~{basename_bam}_below_~{min_coverage}x_coverage.bedgraph
	if [[ "~{histograms}" = "true" ]]
	then
		bedtools genomecov -ibam sorted_u_~{basename_bam}.bam > histogram.txt
	fi
	wget https://raw.githubusercontent.com/lilymaryam/parsevcf/4f75a07b3babfc5c9e0439430925de48171a8fc7/vcf_to_diff_script.py
	python3 vcf_to_diff_script.py -v ~{vcf} -d . -tbmf ~{tbmf} -cf ~{basename_bam}_below_~{min_coverage}x_coverage.bedgraph -cd ~{min_coverage}
	ls -lha
	>>>

	runtime {
		cpu: cpu
		docker: "ashedpotatoes/sranwrp:1.1.6"
		disks: "local-disk " + finalDiskSize + " HDD"
		maxRetries: "${retries}"
		memory: "${memory} GB"
		preemptible: "${preempt}"
	}

	meta {
		author: "Lily Karim (WDLization by Ash O'Farrell)"
	}

	output {
		File diff = basename_bam+".diff"
		File report = basename_bam+".report"
		File mask_file = basename_bam+"_below_"+min_coverage+"x_coverage.bedgraph"
		File? histogram = "histogram.txt"
	}
}

task make_diff {
	input {
		File vcf
		File tbmf
		File cf
		Int cd = 10

		# runtime attributes
		Int addldisk = 10
		Int cpu	= 8
		Int retries	= 1
		Int memory = 16
		Int preempt	= 1
	}
	# estimate disk size
	String basename = basename(vcf)
	Int finalDiskSize = 2*ceil(size(vcf, "GB")) + addldisk

	command <<<
		set -eux pipefail
		mkdir outs
		wget https://raw.githubusercontent.com/lilymaryam/parsevcf/4f75a07b3babfc5c9e0439430925de48171a8fc7/vcf_to_diff_script.py
		python3.10 vcf_to_diff_script.py -v ~{vcf} -d ./outs/ -tbmf ~{tbmf} -cf ~{cf} -cd ~{cd}
		ls -lha outs/
	>>>

	runtime {
		cpu: cpu
		disks: "local-disk " + finalDiskSize + " SSD"
		docker: "ashedpotatoes/sranwrp:1.1.6"
		maxRetries: "${retries}"
		memory: "${memory} GB"
		preemptible: "${preempt}"
	}

	meta {
		author: "Lily Karim (WDLization by Ash O'Farrell)"
	}

	output {
		File diff = "outs/"+basename+".diff"
		File report = "outs/"+basename+".report"
	}
}
