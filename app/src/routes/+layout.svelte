<script>
	import '$lib/styles/app.scss';
	import '$lib/assets/css/prism.css';
	import '$lib/assets/js/prism.js';

	import Header from '$lib/components/Header.svelte';
</script>

<svelte:head>
	<link
		rel="stylesheet"
		href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"
		integrity="sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn"
		crossorigin="anonymous"
	/>
</svelte:head>

<Header />

<main>
	<slot />
</main>
