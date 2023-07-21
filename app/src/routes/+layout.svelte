<script>
	import '$lib/styles/app.scss';
	import '$lib/assets/css/prism.css';
	import '$lib/assets/js/prism.js';

	import Header from '$lib/components/Header.svelte';

	import { StarRegular } from 'svelte-awesome-icons';
	let theme = 'light';
	const toggleTheme = () => (theme == 'light' ? (theme = 'dark') : (theme = 'light'));
</script>

<svelte:head>
	<link
		rel="stylesheet"
		href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"
		integrity="sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn"
		crossorigin="anonymous"
	/>
</svelte:head>

<div class={theme}>
	<div class="app">
		<Header />
		<span on:click={toggleTheme}>
			<StarRegular />
		</span>
		<main>
			<slot />
		</main>
	</div>
</div>

<style>
	.dark {
		--background-color: var(--dark-background);
		--primary-color: var(--dark-primary-color);
		--secondary-color: var(--dark-secondary-color);
		padding: 0;
		margin: 0;
		width: 100%;
		height: 100%;
	}

	.light {
		--background-color: var(--light-background);
		--primary-color: var(--light-primary-color);
		--secondary-color: var(--light-secondary-color);
		padding: 0;
		margin: 0;
		width: 100%;
		height: 100%;
	}

	.app {
		background-color: var(--background-color);
		color: var(--primary-color);
		padding: 0;
		margin: 0;
		width: 100%;
		min-height: 100%;
	}

	main {
		padding: 16px;
		margin: auto;
		max-width: 720px;
	}

	.katex {
		font: 1em sans serif;
	}
</style>
