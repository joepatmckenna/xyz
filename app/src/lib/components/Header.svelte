<script>
	import SlideToggle from '$lib/components/SlideToggle.svelte';
</script>

<header>
	<nav>
		<a href="/">Home</a>
		<a href="/about">About</a>
	</nav>
	<SlideToggle />
</header>

<style>
	header {
		padding: 1rem;
		border-bottom: 4mm ridge rgba(22, 63, 139, 0.6);
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
	}

	a {
		text-decoration: none;
		color: inherit;
	}
</style>
