<script>
	import DarkModeButton from '$lib/components/DarkModeButton.svelte';
</script>

<header>
	<nav>
		<a href="/">Home</a>
		<a href="/about">About</a>
	</nav>
	<DarkModeButton />
</header>

<style>
	header {
		padding: 1rem;
		border-bottom: 2mm ridge var(--secondary-color);
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
	}

	a {
		text-decoration: none;
		color: inherit;
		padding: 8px;
	}
</style>
