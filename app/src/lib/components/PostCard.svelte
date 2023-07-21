<script>
	export let post;
	export let index;
</script>

<div class="container">
	<div class="header">
		<div class="title">
			<a href={post.path}>{post.metadata.title}</a>
		</div>
		<div class="subtitle">
			{post.metadata.date}
		</div>
	</div>
	<div class="body">
		{#if index == 0}
			<!-- {@html post.html} -->
		{/if}
	</div>
</div>

<style>
	a:link,
	a:visited {
		text-decoration: none;
		color: var(--primary-color);
	}

	a:hover {
		color: var(--secondary-color);
	}

	.container {
		padding: 8px;
		margin: auto;
	}

	.header {
		padding: 4px;
	}

	.title {
		font-size: 1.1em;
	}

	.subtitle {
		font-size: 0.9em;
	}
</style>
