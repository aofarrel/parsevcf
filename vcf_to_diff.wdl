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
		Int min_coverage_per_site
		Boolean diffs = true
		Boolean histograms = false

		# runtime attributes
		Int addldisk = 10
		Int cpu      = 8
		Int retries  = 1
		Int memory   = 16
		Int preempt  = 1
	}
	String basename_bam = basename(bam, ".bam")
	String basename_vcf = basename(vcf, ".vcf")
	Int finalDiskSize = ceil(size(bam, "GB")*2) + ceil(size(vcf, "GB")*2) + addldisk
	
	command <<<
	set -eux pipefail
	start=$(date +%s)
	echo "Copying bam..."
	cp ~{bam} .
	echo "Sorting bam..."
	samtools sort -u ~{basename_bam}.bam > sorted_u_~{basename_bam}.bam
	echo "Calculating coverage..."
	bedtools genomecov -ibam sorted_u_~{basename_bam}.bam -bga | \
		awk '$4 < ~{min_coverage_per_site}' > \
		~{basename_bam}_below_~{min_coverage_per_site}x_coverage.bedgraph
	if [[ "~{histograms}" = "true" ]]
	then
		echo "Generating histograms..."
		bedtools genomecov -ibam sorted_u_~{basename_bam}.bam > histogram.txt
	fi
	if [[ "~{diffs}" = "true" ]]
	then
		echo "Pulling script..."
		wget https://raw.githubusercontent.com/aofarrel/parsevcf/1.1.5/vcf_to_diff_script.py
		echo "Running script..."
		python3 vcf_to_diff_script.py -v ~{vcf} \
		-d . \
		-tbmf ~{tbmf} \
		-bam ~{basename_bam}_below_~{min_coverage_per_site}x_coverage.bedgraph \
		-cd ~{min_coverage_per_site}
	fi
	end=$(date +%s)
	seconds=$(echo "$end - $start" | bc)
	minutes=$(echo "$seconds" / 60 | bc)
	echo "Finished in about $minutes minutes ($seconds sec))"
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
		File mask_file = basename_bam+"_below_"+min_coverage_per_site+"x_coverage.bedgraph"
		File? diff = basename_vcf+".diff"
		File? report = basename_vcf+".report"
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
		wget https://raw.githubusercontent.com/lilymaryam/parsevcf/1.0.4/vcf_to_diff_script.py
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
