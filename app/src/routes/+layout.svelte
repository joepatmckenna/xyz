<script>
	import '$lib/styles/app.scss';
	import '$lib/assets/css/prism.css';
	import '$lib/assets/js/prism.js';
	import Header from '$lib/components/Header.svelte';
	import { darkMode } from '$lib/stores.js';
</script>

<svelte:head>
	<link
		rel="stylesheet"
		href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"
		integrity="sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn"
		crossorigin="anonymous"
	/>
</svelte:head>

<div class="app" theme={$darkMode ? 'dark' : 'light'}>
	<Header />
	<main>
		<slot />
	</main>
</div>

<style>
	.app {
		background-color: var(--background-color);
		color: var(--primary-color);
		padding: 0;
		margin: 0;
		min-width: 100%;
		min-height: 100%;
	}

	main {
		padding: 16px;
		margin: auto;
		max-width: 720px;
	}
</style>
