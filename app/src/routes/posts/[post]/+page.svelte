<script>
	export let data;
	const { title, date, Content } = data;
</script>

<article>
	<h1>{title}</h1>
	<p>{date}</p>
	<Content />
</article>
